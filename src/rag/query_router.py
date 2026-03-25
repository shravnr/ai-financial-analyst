import json
import re
import logging

from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL_MINI

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

# Regex fast path — keyword patterns to source preferences
_PATTERNS: list[tuple[str, list[str], int]] = [
    (r"(?i)(revenue|profit|income|earnings|eps|margin|ebitda)\b.*\b(number|figure|exact|how much|what)",
     ["sec"], 12),

    (r"(?i)(latest|recent|news|today|this week|this month|announce|event)",
     ["sec"], 12),

    (r"(?i)(risk|threat|challenge|headwind|concern|uncertainty|litigation|lawsuit)",
     ["sec"], 12),

    (r"(?i)(guidance|outlook|forecast|forward|strategy|plan|initiative|management said|going forward)",
     ["sec"], 12),

    (r"(?i)(debt|leverage|liabilit|borrow|loan|credit|bond|cash position|cash flow|liquidity|balance sheet)",
     ["sec"], 12),

    (r"(?i)(valuation|overvalued|undervalued|p\/e|pe ratio|price.to|ev\/|enterprise value|fair value)",
     ["sec"], 10),

    (r"(?i)(analyst|rating|upgrade|downgrade|buy|sell|hold|target price|consensus|estimate)",
     ["sec"], 10),

    (r"(?i)(competitor|compar|versus|vs\.?|peer|rival|industry|sector)",
     ["sec"], 12),

    (r"(?i)(segment|division|business line|revenue driver|breakdown|geographic|product mix)",
     ["sec"], 12),

    (r"(?i)(earnings call|earnings report|quarterly result|quarter result|conference call)",
     ["sec"], 12),
]

# LLM fallback for queries that don't match regex

_LLM_ROUTER_SYSTEM = """You are a financial question classifier. Given a user's question about a company, determine how many SEC filing chunks to retrieve.

The vector store contains only SEC filings (10-K annual reports, 10-Q quarterly reports, 8-K earnings press releases). These cover: risks, strategy, business segments, legal matters, management discussion, financial statements, qualitative analysis.

Return JSON with:
- "source_types": ["sec"]
- "n_results": number of chunks to retrieve (8-15, use 12 for broad questions, 8-10 for focused ones)
- "reasoning": one sentence explaining your choice

Example: {"source_types": ["sec"], "n_results": 12, "reasoning": "Revenue comparison needs multiple filing periods for context."}"""


def _llm_classify(question: str) -> dict:
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL_MINI,
            messages=[
                {"role": "system", "content": _LLM_ROUTER_SYSTEM},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=150,
        )

        result = json.loads(response.choices[0].message.content)
        source_types = result.get("source_types")
        n_results = result.get("n_results", 12)
        reasoning = result.get("reasoning", "")

        valid_sources = {"sec", "fmp", "news"}
        if source_types is not None:
            source_types = [s for s in source_types if s in valid_sources]
            if not source_types:
                source_types = None  # fall back to all

        n_results = max(6, min(15, n_results))

        logger.info(f"LLM router: sources={source_types}, n={n_results}, reason={reasoning}")

        return {
            "source_types": source_types,
            "n_results": n_results,
            "matched_pattern": f"llm: {reasoning}",
            "router_type": "llm",
        }

    except Exception as e:
        logger.warning(f"LLM router failed ({e}), falling back to default")
        return {
            "source_types": None,
            "n_results": 12,
            "matched_pattern": "default (llm failed)",
            "router_type": "fallback",
        }


def route_query(question: str) -> dict:
    for pattern, sources, n_results in _PATTERNS:
        if re.search(pattern, question):
            logger.debug(f"Regex matched: {pattern[:50]}... → sources={sources}")
            return {
                "source_types": sources,
                "n_results": n_results,
                "matched_pattern": pattern[:50],
                "router_type": "regex",
            }

    logger.info(f"No regex match for: '{question[:80]}' — escalating to LLM router")
    return _llm_classify(question)
