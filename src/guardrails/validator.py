import json
import re
import logging

from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL_MINI

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)


def _check_citations(answer: str) -> list[str]:
    warnings = []
    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]

    for para in paragraphs:
        # Skip short paragraphs (transitions, headers)
        if len(para) < 80:
            continue

        # Does this paragraph contain numbers/facts?
        has_numbers = bool(re.search(
            r"\$[\d,.]+|\d+\.?\d*\s*(%|percent|billion|million|trillion)", para
        ))

        # Does it have a citation?
        has_citation = bool(re.search(r"\[\d+\]", para))

        # Does it have refusal language? (exempt from citation requirement)
        is_refusal = bool(re.search(
            r"(?i)(don't have enough|not available|insufficient|cannot determine|"
            r"no .* data|not .* in the .* context)", para
        ))

        if has_numbers and not has_citation and not is_refusal:
            # Extract the first number for the warning
            number_match = re.search(r"(\$[\d,.]+\s*\w*|\d+\.?\d*\s*%)", para)
            number_str = number_match.group(0) if number_match else "a number"
            warnings.append(
                f"Uncited claim: paragraph mentions {number_str} without a [Source N] citation."
            )

    return warnings


def _check_numbers_in_context(answer: str, context: str) -> list[str]:
    warnings = []

    # Extract all dollar amounts from context for numeric comparison
    context_amounts = set()
    for m in re.finditer(r"\$([\d,.]+)\s*(billion|million|trillion|thousand|B|M|T|K)?", context):
        try:
            val = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").lower()
            if unit in ("billion", "b"):
                val *= 1e9
            elif unit in ("million", "m"):
                val *= 1e6
            elif unit in ("trillion", "t"):
                val *= 1e12
            elif unit in ("thousand", "k"):
                val *= 1e3
            context_amounts.add(val)
        except ValueError:
            pass
    # Also extract raw large numbers from context (e.g., from SEC filings)
    for m in re.finditer(r"[\$]?([\d,]{10,})", context):
        try:
            context_amounts.add(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    # Extract dollar amounts from the answer
    answer_amounts = re.findall(r"\$([\d,.]+)\s*(billion|million|trillion|thousand|B|M|T|K)?", answer)

    for amount_str, unit in answer_amounts:
        clean = amount_str.replace(",", "")
        found = False

        # 1. Direct string match (exact)
        if f"${amount_str}" in context or f"${clean}" in context:
            found = True

        # 2. Number string without dollar sign
        if not found and (clean in context or amount_str in context):
            found = True

        # 3. Numeric comparison with tolerance
        if not found:
            try:
                val = float(clean)
                u = (unit or "").lower()
                if u in ("billion", "b"):
                    val *= 1e9
                elif u in ("million", "m"):
                    val *= 1e6
                elif u in ("trillion", "t"):
                    val *= 1e12
                elif u in ("thousand", "k"):
                    val *= 1e3

                # Check within 1% tolerance of any context amount
                for ctx_val in context_amounts:
                    if ctx_val == 0:
                        continue
                    if abs(val - ctx_val) / abs(ctx_val) < 0.01:
                        found = True
                        break

                # Also try the raw integer form in context string
                if not found:
                    raw = f"{val:.0f}"
                    if raw in context:
                        found = True
                    # Try with commas: 394,328,000,000
                    raw_comma = f"{val:,.0f}"
                    if raw_comma in context:
                        found = True

                # 4. Check if number is derivable from context numbers
                if not found:
                    ctx_list = list(context_amounts)
                    for i, a in enumerate(ctx_list):
                        for b in ctx_list[i + 1:]:
                            for derived in (a - b, b - a, a + b):
                                if derived > 0 and abs(val - derived) / derived < 0.01:
                                    found = True
                                    break
                            if not found and b != 0:
                                for ratio in (a / b, b / a):
                                    if ratio > 0 and abs(val - ratio) / ratio < 0.01:
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
            except ValueError:
                pass

        if not found:
            warnings.append(
                f"Unverified number: ${amount_str} {unit or ''} appears in the answer "
                f"but could not be matched in the retrieved context."
            )

    return warnings


def _check_sufficiency(answer: str, chunk_count: int) -> list[str]:
    warnings = []

    if chunk_count == 0:
        return warnings  # No data exists at all; answer is inherently about that

    if chunk_count <= 2:
        has_caveat = bool(re.search(
            r"(?i)(limited|insufficient|don't have enough|not available|"
            r"based on .* available data|only .* source)", answer
        ))
        if not has_caveat:
            warnings.append(
                "Low context: only {count} chunk(s) retrieved but answer does not "
                "acknowledge data limitations.".format(count=chunk_count)
            )

    return warnings


#  LLM Grounding Check

_GROUNDING_SYSTEM = """You are a fact-checking assistant. Given an AI-generated answer about a company's finances and the source context that was provided to the AI, check whether key factual claims in the answer are actually supported by the sources.

Extract the top 3 most important factual claims (numbers, dates, or specific statements) from the answer. For each claim, check if the provided context supports it.

Return JSON:
{
  "claims": [
    {
      "claim": "the factual claim extracted from the answer",
      "supported": true/false,
      "evidence": "brief quote or reference from context that supports/contradicts, or 'not found in context'"
    }
  ]
}

Be strict: a claim is only "supported" if the context contains information that directly backs it. Reasonable rounding (e.g., $393.9B reported as ~$394B) is acceptable. But a claim with no basis in the context is NOT supported."""


def _llm_grounding_check(answer: str, context: str) -> list[str]:
    # Skip if answer is very short (likely a refusal)
    if len(answer) < 100:
        return []

    # Truncate context to fit in mini's context window cost-effectively
    ctx_truncated = context[:8000] if len(context) > 8000 else context

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL_MINI,
            messages=[
                {"role": "system", "content": _GROUNDING_SYSTEM},
                {"role": "user", "content": f"ANSWER:\n{answer}\n\nCONTEXT:\n{ctx_truncated}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500,
        )

        result = json.loads(response.choices[0].message.content)
        warnings = []

        for claim_check in result.get("claims", []):
            if not claim_check.get("supported", True):
                claim = claim_check.get("claim", "unknown claim")
                evidence = claim_check.get("evidence", "no evidence found")
                warnings.append(
                    f"Grounding issue: \"{claim}\" — {evidence}"
                )

        if warnings:
            logger.info(f"LLM grounding found {len(warnings)} unsupported claim(s)")

        return warnings

    except Exception as e:
        logger.warning(f"LLM grounding check failed ({e}), skipping")
        return []  # graceful degradation — don't block on validation failure


def validate_response(
    answer: str,
    context: str,
    chunk_count: int,
    use_llm_grounding: bool = True,
) -> dict:
    checks = {
        "citation_check": _check_citations(answer),
        "number_verification": _check_numbers_in_context(answer, context),
        "sufficiency_check": _check_sufficiency(answer, chunk_count),
    }

    if use_llm_grounding:
        checks["grounding_check"] = _llm_grounding_check(answer, context)

    all_warnings = []
    for check_warnings in checks.values():
        all_warnings.extend(check_warnings)

    if all_warnings:
        logger.warning(f"Guardrail warnings ({len(all_warnings)}): {all_warnings}")

    return {
        "is_valid": len(all_warnings) == 0,
        "warnings": all_warnings,
        "checks": checks,
    }
