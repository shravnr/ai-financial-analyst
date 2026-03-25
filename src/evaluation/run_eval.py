import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI

from src.config import OPENAI_API_KEY, EVAL_MODEL, LLM_MODEL_MINI
from src.guardrails.validator import verify_numbers
from src.rag.chain import ask
from src.evaluation.test_questions import (
    CONSISTENCY_PAIRS,
    GROUNDING_QUESTIONS,
    SMOKE_QUESTIONS,
    BOUNDARY_REFUSALS,
)
from src.processing.vector_store import is_ticker_indexed
from src.ingestion.orchestrator import ingest_company
from src.processing.pipeline import process_company

logger = logging.getLogger(__name__)

EVAL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
DEEP_EVAL_TICKER = "AMZN"

# Ticker → company name (used to resolve {company} in question templates)
_TICKER_COMPANY = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "TSLA": "Tesla", "META": "Meta", "NVDA": "NVIDIA",
    "GOOG": "Alphabet", "NFLX": "Netflix", "JPM": "JPMorgan",
}


def _company_name(ticker: str) -> str:
    return _TICKER_COMPANY.get(ticker.upper(), ticker)

_client = OpenAI(api_key=OPENAI_API_KEY)



def _count_citations(text: str) -> int:
    return len(re.findall(r"\[\d+\]", text))


def _has_dollar_amounts(text: str) -> bool:
    return bool(re.search(r"\$[\d,.]+\s*(billion|million|trillion|B|M)?", text))



_GROUNDING_SYSTEM = """You are an expert evaluator assessing whether an AI financial analyst's answer is factually grounded in the retrieved source data.

Given:
- The user's QUESTION
- The AI's ANSWER (with [Source N] citations)
- The CONTEXT (source data that was available to the AI)

Score the answer's factual grounding from 1-5:

1 = FABRICATED: Answer contains claims that contradict or have no basis in the context
2 = MOSTLY UNSUPPORTED: Most claims lack support in the context
3 = PARTIALLY GROUNDED: Some claims are supported, others are not or are stretched
4 = WELL GROUNDED: Most claims are directly supported by context, minor gaps acceptable
5 = FULLY GROUNDED: Every factual claim in the answer is clearly supported by the context

Scoring guidelines:
- Reasonable rounding ($393.9B → ~$394B) is acceptable, score 4-5
- Correct refusal ("I don't have enough information") when context is thin scores 5
- A well-cited answer that misrepresents what sources say scores 1-2
- Focus on factual accuracy, not writing quality or completeness

Return JSON:
{
  "score": <1-5>,
  "reasoning": "<2-3 sentences explaining the score>"
}"""


_CONSISTENCY_SYSTEM = """You are evaluating whether two AI-generated answers to semantically equivalent financial questions are consistent with each other.

Given:
- QUESTION_A and ANSWER_A
- QUESTION_B and ANSWER_B

Score consistency from 1-5:

1 = CONTRADICTORY: Answers give conflicting facts, figures, or conclusions
2 = MOSTLY INCONSISTENT: Key claims or numbers differ significantly
3 = PARTIALLY CONSISTENT: Core facts agree but notable differences in details or figures
4 = CONSISTENT: Same key facts and figures, minor wording or emphasis differences
5 = FULLY CONSISTENT: Essentially the same answer expressed differently

Scoring guidelines:
- Focus on factual consistency (numbers, claims, conclusions), not phrasing
- Minor rounding differences ($391B vs ~$390B) are acceptable, score 4-5
- One answer having extra detail is fine as long as shared facts agree
- Contradictory numbers or opposite conclusions score 1-2

Return JSON:
{
  "score": <1-5>,
  "reasoning": "<2-3 sentences explaining the score>"
}"""


_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles each retry


def _call_judge(messages: list[dict], label: str = "judge", model: str = EVAL_MODEL) -> dict:
    """Call the eval LLM with retry on rate-limit (429) errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=200,
            )
            result = json.loads(response.choices[0].message.content)
            return {
                "score": result.get("score", 0),
                "reasoning": result.get("reasoning", ""),
                "tokens": response.usage.total_tokens,
                **{k: v for k, v in result.items() if k not in ("score", "reasoning")},
            }
        except Exception as e:
            if "429" in str(e) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"{label} rate-limited, retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            logger.error(f"{label} failed: {e}")
            return {"score": 0, "reasoning": f"Judge error: {e}", "tokens": 0}


def _judge_grounding(question: str, answer: str, context: str) -> dict:
    ctx_truncated = context[:12000] if len(context) > 12000 else context
    return _call_judge(
        messages=[
            {"role": "system", "content": _GROUNDING_SYSTEM},
            {"role": "user", "content": (
                f"QUESTION: {question}\n\n"
                f"ANSWER: {answer}\n\n"
                f"CONTEXT:\n{ctx_truncated}"
            )},
        ],
        label="grounding",
    )


def _judge_consistency(q_a: str, a_a: str, q_b: str, a_b: str) -> dict:
    return _call_judge(
        messages=[
            {"role": "system", "content": _CONSISTENCY_SYSTEM},
            {"role": "user", "content": (
                f"QUESTION_A: {q_a}\n\n"
                f"ANSWER_A: {a_a}\n\n"
                f"QUESTION_B: {q_b}\n\n"
                f"ANSWER_B: {a_b}"
            )},
        ],
        label="consistency",
    )


_REFUSAL_SYSTEM = """You are classifying whether an AI financial analyst's answer refuses, declines, or acknowledges inability to provide the requested information.

A refusal includes:
- Explicitly saying it cannot, could not, or does not have the data
- Acknowledging the information is unavailable, not public, or out of scope
- Declining to predict or speculate

A refusal does NOT include:
- Providing actual financial data (even if partial)
- Offering to help with related questions (this can accompany a refusal)

Return JSON:
{
  "is_refusal": true/false,
  "reasoning": "<1 sentence>"
}"""


def _judge_refusal(answer: str) -> dict:
    result = _call_judge(
        messages=[
            {"role": "system", "content": _REFUSAL_SYSTEM},
            {"role": "user", "content": f"ANSWER:\n{answer}"},
        ],
        label="refusal",
        model=LLM_MODEL_MINI,
    )
    return {
        "is_refusal": result.get("is_refusal", False),
        "reasoning": result.get("reasoning", ""),
        "tokens": result.get("tokens", 0),
    }



def run_deep_eval(ticker: str = "AAPL") -> dict:
    ticker = ticker.upper()

    print(f"\n{'='*60}")
    print(f"  Deep Eval: {ticker}")
    print(f"  Signals: Consistency (4) + Grounding (10) + Accuracy (10)")
    print(f"  Judge calls: ~14 (4 consistency + 10 grounding)")
    print(f"{'='*60}\n")

    consistency_results = []
    grounding_results = []
    total_pipeline_tokens = 0
    total_judge_tokens = 0

    #  Signal 1: Consistency 
    print(f"---- Signal 1: CONSISTENCY \n")

    company = _company_name(ticker)

    for i, pair in enumerate(CONSISTENCY_PAIRS, 1):
        name = pair["name"]
        q_a, q_b = [q.format(company=company) for q in pair["questions"]]
        print(f"  [{i}/{len(CONSISTENCY_PAIRS)}] {name}...")

        try:
            resp_a = ask(q_a, ticker, show_context=True)
            resp_b = ask(q_b, ticker, show_context=True)
            a_a = resp_a["answer"]
            a_b = resp_b["answer"]

            tokens_a = resp_a.get("token_usage", {}).get("total_tokens", 0)
            tokens_b = resp_b.get("token_usage", {}).get("total_tokens", 0)
            total_pipeline_tokens += tokens_a + tokens_b

            judge = _judge_consistency(q_a, a_a, q_b, a_b)
            total_judge_tokens += judge.get("tokens", 0)

            passed = judge["score"] >= 4
            print(f"      → score={judge['score']}/5  {'PASS' if passed else 'FAIL'}")

            consistency_results.append({
                "name": name,
                "question_a": q_a,
                "question_b": q_b,
                "answer_a": a_a,
                "answer_b": a_b,
                "score": judge["score"],
                "reasoning": judge["reasoning"],
                "passed": passed,
                "pipeline_tokens": tokens_a + tokens_b,
                "judge_tokens": judge.get("tokens", 0),
            })

        except Exception as e:
            logger.error(f"Consistency test failed for {name}: {e}")
            print(f"      → ERROR: {e}  FAIL")
            consistency_results.append({
                "name": name,
                "question_a": q_a,
                "question_b": q_b,
                "score": 0,
                "reasoning": f"Error: {e}",
                "passed": False,
            })

    #  Signal 2 & 3: Grounding + Accuracy 
    print(f"\n---- Signal 2: GROUNDING + Signal 3: ACCURACY \n")

    for i, (capability, question_tpl, _) in enumerate(GROUNDING_QUESTIONS, 1):
        question = question_tpl.format(company=company)
        print(f"  [{i}/{len(GROUNDING_QUESTIONS)}] {capability}...")

        try:
            response = ask(question, ticker, show_context=True)
            answer = response["answer"]
            context = response.get("context", "")

            tokens = response.get("token_usage", {}).get("total_tokens", 0)
            total_pipeline_tokens += tokens

            citations = _count_citations(answer)
            num_check = verify_numbers(answer, context)

            judge = _judge_grounding(question, answer, context)
            total_judge_tokens += judge.get("tokens", 0)

            grounding_passed = judge["score"] >= 4
            has_citations = citations > 0
            all_nums_verified = num_check["total"] == 0 or len(num_check["unverified_amounts"]) == 0

            status_parts = [
                f"ground={judge['score']}/5",
                f"{citations} cites",
                f"nums={num_check['verified']}/{num_check['total']}",
            ]
            overall = "PASS" if (grounding_passed and has_citations and all_nums_verified) else "FAIL"
            print(f"      → {', '.join(status_parts)}  {overall}")

            grounding_results.append({
                "capability": capability,
                "question": question,
                "answer": answer,
                "grounding_score": judge["score"],
                "grounding_reasoning": judge["reasoning"],
                "grounding_passed": grounding_passed,
                "citations": citations,
                "has_citations": has_citations,
                "numbers_total": num_check["total"],
                "numbers_verified": num_check["verified"],
                "pipeline_tokens": tokens,
                "judge_tokens": judge.get("tokens", 0),
            })

        except Exception as e:
            logger.error(f"Grounding test failed for {capability}: {e}")
            print(f"      → ERROR: {e}  FAIL")
            grounding_results.append({
                "capability": capability,
                "question": question,
                "grounding_score": 0,
                "grounding_reasoning": f"Error: {e}",
                "grounding_passed": False,
                "citations": 0,
                "has_citations": False,
                "numbers_total": 0,
                "numbers_verified": 0,
            })

    #  Aggregate stats for return dict 
    cons_passed = sum(1 for r in consistency_results if r["passed"])
    cons_scores = [r["score"] for r in consistency_results if r["score"] > 0]
    avg_cons = sum(cons_scores) / len(cons_scores) if cons_scores else 0

    ground_passed = sum(1 for r in grounding_results if r["grounding_passed"])
    ground_scores = [r["grounding_score"] for r in grounding_results if r["grounding_score"] > 0]
    avg_ground = sum(ground_scores) / len(ground_scores) if ground_scores else 0
    well_grounded = sum(1 for s in ground_scores if s >= 4)

    cited = sum(1 for r in grounding_results if r["has_citations"])
    total_nums = sum(r["numbers_total"] for r in grounding_results)
    verified_nums = sum(r["numbers_verified"] for r in grounding_results)

    total_cons = len(CONSISTENCY_PAIRS)
    total_ground = len(GROUNDING_QUESTIONS)

    return {
        "ticker": ticker,
        "signals": {
            "consistency": {
                "passed": cons_passed,
                "total": total_cons,
                "avg_score": round(avg_cons, 2),
                "details": consistency_results,
            },
            "grounding": {
                "passed": ground_passed,
                "total": total_ground,
                "avg_score": round(avg_ground, 2),
                "well_grounded": well_grounded,
                "details": grounding_results,
            },
            "accuracy": {
                "citations_present": cited,
                "numbers_verified": verified_nums,
                "numbers_total": total_nums,
                "total": total_ground,
            },
        },
        "cost": {
            "pipeline_tokens": total_pipeline_tokens,
            "judge_tokens": total_judge_tokens,
            "total_tokens": total_pipeline_tokens + total_judge_tokens,
        },
    }



def _ensure_indexed(ticker: str) -> bool:
    if is_ticker_indexed(ticker):
        return True
    print(f"\n  ⏳ {ticker} not indexed yet — fetching & indexing now (≈30-60s, one-time only)...\n")
    try:
        ingest_company(ticker)
        process_company(ticker)
        return is_ticker_indexed(ticker)
    except Exception as e:
        logger.error(f"Failed to ingest {ticker}: {e}")
        return False



def run_smoke_test(tickers: list[str] | None = None) -> dict:

    smoke_tickers = [t.upper() for t in (tickers or [])]

    print(f"\n{'='*60}")
    print(f"  Smoke Test")
    if smoke_tickers:
        print(f"  Tickers: {', '.join(smoke_tickers)}")
        print(f"  Questions: {len(SMOKE_QUESTIONS)} per ticker + {len(BOUNDARY_REFUSALS)} boundary refusals")
    else:
        print(f"  Tickers: none")
        print(f"  Questions: {len(BOUNDARY_REFUSALS)} boundary refusals only")
    print(f"  Judge calls: {len(BOUNDARY_REFUSALS)} (boundary refusal classifier)")
    print(f"{'='*60}\n")

    ticker_results = {}
    boundary_results = []
    total_tokens = 0

    #  Normal questions per ticker 
    for ticker in smoke_tickers:
        ticker = ticker.upper()
        print(f"   {ticker} \n")

        results = []
        for i, (capability, question_template, _) in enumerate(SMOKE_QUESTIONS, 1):
            question = question_template.format(ticker=ticker)
            print(f"  [{i}/{len(SMOKE_QUESTIONS)}] {capability}...")

            try:
                response = ask(question, ticker, show_context=True)
                answer = response["answer"]
                tokens = response.get("token_usage", {}).get("total_tokens", 0)
                total_tokens += tokens

                citations = _count_citations(answer)
                source_count = len(response.get("sources", []))
                passed = citations > 0
                thin = source_count <= 1
                print(f"      → {citations} cites, {source_count} sources{'  ⚠ thin' if thin else ''}  {'PASS' if passed else 'FAIL'}")

                results.append({
                    "capability": capability,
                    "question": question,
                    "citations": citations,
                    "source_count": source_count,
                    "thin_context": thin,
                    "passed": passed,
                    "tokens": tokens,
                })

            except Exception as e:
                logger.error(f"Smoke test failed for {ticker}/{capability}: {e}")
                print(f"      → ERROR: {e}  FAIL")
                results.append({
                    "capability": capability,
                    "question": question,
                    "citations": 0,
                    "passed": False,
                    "error": str(e),
                })

        ticker_results[ticker] = results
        print()

    #  Boundary refusals 
    print(f"---- Boundary refusals \n")

    for i, (name, question) in enumerate(BOUNDARY_REFUSALS, 1):
        print(f"  [{i}/{len(BOUNDARY_REFUSALS)}] {name}...")

        try:
            # Use a dummy ticker for boundary cases
            response = ask(question, "UNKNOWN", show_context=True)
            answer = response["answer"]
            tokens = response.get("token_usage", {}).get("total_tokens", 0)
            total_tokens += tokens

            has_dollars = _has_dollar_amounts(answer)
            refusal = _judge_refusal(answer)
            has_refusal = refusal["is_refusal"]
            total_tokens += refusal.get("tokens", 0)
            # Pass: no fabricated dollar amounts AND classified as refusal
            passed = not has_dollars and has_refusal
            status = "PASS" if passed else "FAIL"
            detail = f"dollars={has_dollars}, refusal={has_refusal}"
            print(f"      → {detail}  {status}")

            boundary_results.append({
                "name": name,
                "question": question,
                "answer": answer,
                "has_dollar_amounts": has_dollars,
                "has_refusal": has_refusal,
                "refusal_reasoning": refusal["reasoning"],
                "passed": passed,
                "tokens": tokens,
            })

        except Exception as e:
            logger.error(f"Boundary test failed for {name}: {e}")
            print(f"      → ERROR: {e}  FAIL")
            boundary_results.append({
                "name": name,
                "question": question,
                "passed": False,
                "error": str(e),
            })

    bp = sum(1 for r in boundary_results if r["passed"])
    bt = len(boundary_results)

    return {
        "tickers": list(ticker_results.keys()),
        "ticker_results": {
            t: {"passed": sum(1 for r in rs if r["passed"]), "total": len(rs), "details": rs}
            for t, rs in ticker_results.items()
        },
        "boundary_refusals": {
            "passed": bp,
            "total": bt,
            "details": boundary_results,
        },
        "cost": {"total_tokens": total_tokens},
    }



def _print_scorecard(deep: dict, smoke: dict) -> dict:
    d = deep["signals"]

    # Sub-score inputs
    avg_cons = d["consistency"]["avg_score"]
    avg_ground = d["grounding"]["avg_score"]
    cited = d["accuracy"]["citations_present"]
    nums_verified = d["accuracy"]["numbers_verified"]
    nums_total = d["accuracy"]["numbers_total"]
    total_ground = d["accuracy"]["total"]
    cons_passed = d["consistency"]["passed"]
    total_cons = d["consistency"]["total"]
    well_grounded = d["grounding"]["well_grounded"]

    smoke_passed = sum(
        r["passed"] for tr in smoke["ticker_results"].values() for r in tr["details"]
    )
    smoke_total = sum(tr["total"] for tr in smoke["ticker_results"].values())
    boundary_passed = smoke["boundary_refusals"]["passed"]
    boundary_total = smoke["boundary_refusals"]["total"]

    # Normalize to 1-10
    consistency_score = round((avg_cons / 5) * 10, 1)
    grounding_score = round((avg_ground / 5) * 10, 1)
    citations_pct = cited / total_ground if total_ground else 0
    numbers_pct = nums_verified / nums_total if nums_total else 1.0
    verification_score = round(((citations_pct + numbers_pct) / 2) * 10, 1)
    has_smoke = smoke_total > 0
    smoke_score = round((smoke_passed / smoke_total) * 10, 1) if has_smoke else None
    boundary_score = round((boundary_passed / boundary_total) * 10, 1) if boundary_total else 0

    # Group scores by method
    llm_score = round((consistency_score + grounding_score + boundary_score) / 3, 1)
    if has_smoke:
        rule_score = round((verification_score + smoke_score) / 2, 1)
    else:
        rule_score = verification_score
    master = round((llm_score + rule_score) / 2, 1)

    total_tokens = deep["cost"]["total_tokens"] + smoke["cost"]["total_tokens"]

    # Print
    print(f"\n{'═'*60}")
    print(f"  EVAL SCORECARD")
    print(f"{'═'*60}\n")
    print(f"  Master Score:        {master} / 10\n")

    print(f"   LLM-as-judge (avg: {llm_score}) ")
    print(f"  Consistency:  {consistency_score:>4} / 10   ({cons_passed}/{total_cons} pairs, avg {avg_cons:.2f}/5)")
    print(f"  Grounding:    {grounding_score:>4} / 10   ({well_grounded}/{total_ground} well-grounded, avg {avg_ground:.2f}/5)")
    print(f"  Boundary:     {boundary_score:>4} / 10   ({boundary_passed}/{boundary_total} refusals)")

    print(f"\n   Rule-based (avg: {rule_score}) ")
    print(f"  Verification: {verification_score:>4} / 10   ({cited}/{total_ground} cited, {nums_verified}/{nums_total} nums verified)")
    if has_smoke:
        print(f"  Smoke:        {smoke_score:>4} / 10   ({smoke_passed}/{smoke_total} passed)")
    else:
        print(f"  Smoke:         N/A        (no additional tickers indexed)")


    # Failures
    failures = []
    for r in d["consistency"]["details"]:
        if not r["passed"]:
            failures.append(f"  Consistency: {r['name']} (score {r['score']})")
    for r in d["grounding"]["details"]:
        if not r["grounding_passed"]:
            failures.append(f"  Grounding: {r['capability']} (score {r['grounding_score']})")
    for r in smoke["boundary_refusals"]["details"]:
        if not r["passed"]:
            failures.append(f"  Boundary: {r['name']} (no refusal detected)")
    # Thin context warnings from smoke
    for tr in smoke["ticker_results"].values():
        for r in tr["details"]:
            if r.get("thin_context"):
                failures.append(f"  Smoke: {r['question'][:40]}... ({r['source_count']} source(s) \u2014 thin data)")

    if failures:
        print(f"\n   Failures & Flags: ")
        for f in failures:
            print(f"  {f}")

    verdict = "PASS" if master >= 8.5 else "FAIL"
    verdict_icon = "\u2705" if master >= 8.5 else "\u274c"

    print(f"\n  Cost: {total_tokens:,} tokens")
    print(f"\n  Verdict:             {verdict} {verdict_icon}  ({'\u2265' if master >= 8.5 else '<'} 8.5)")
    print(f"{'═'*60}\n")

    return {
        "master_score": master,
        "verdict": verdict,
        "sub_scores": {
            "llm_judge": llm_score,
            "rule_based": rule_score,
            "consistency": consistency_score,
            "grounding": grounding_score,
            "boundary": boundary_score,
            "verification": verification_score,
            "smoke": smoke_score,
        },
    }



if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    import random

    # Ensure deep eval ticker is indexed
    _ensure_indexed(DEEP_EVAL_TICKER)
    deep = run_deep_eval(DEEP_EVAL_TICKER)

    # Pick a random smoke ticker (not the deep eval ticker), auto-ingest if needed
    smoke_candidates = [t for t in EVAL_TICKERS if t != DEEP_EVAL_TICKER]
    smoke_ticker = random.choice(smoke_candidates)

    _ensure_indexed(smoke_ticker)
    smoke = run_smoke_test(tickers=[smoke_ticker])

    scores = _print_scorecard(deep, smoke)

    #  Save combined results 
    eval_dir = project_root / "eval_results"
    eval_dir.mkdir(exist_ok=True)
    output_file = eval_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        **scores,
        "deep_eval": deep,
        "smoke_test": smoke,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Results saved to: {output_file}")
