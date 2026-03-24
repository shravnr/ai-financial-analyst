import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI

from src.config import OPENAI_API_KEY, EVAL_MODEL
from src.rag.chain import ask
from src.evaluation.test_questions import AAPL_TEST_QUESTIONS

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)


# ── Rule-based metrics ───────────────────────────────────────────────

def _count_citations(text: str) -> int:
    return len(re.findall(r"\[\d+\]", text))


def _has_refusal(text: str) -> bool:
    return bool(re.search(
        r"(?i)(don't have enough information|not available in the|"
        r"insufficient .* data|cannot .* from the available|"
        r"not .* in the retrieved context)", text
    ))


def _has_dollar_amounts(text: str) -> bool:
    return bool(re.search(r"\$[\d,.]+\s*(billion|million|trillion|B|M)?", text))


# ── LLM-as-Judge grounding scorer ───────────────────────────────────

_JUDGE_SYSTEM = """You are an expert evaluator assessing whether an AI financial analyst's answer is factually grounded in the retrieved source data.

Given:
- The user's QUESTION
- The AI's ANSWER (with [Source N] citations)
- The CONTEXT (source data that was available to the AI)
- A GROUND TRUTH reference (approximate expected answer)

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


def _judge_grounding(
    question: str,
    answer: str,
    context: str,
    ground_truth: str,
) -> dict:
    # Truncate context to manage cost
    ctx_truncated = context[:6000] if len(context) > 6000 else context

    try:
        response = _client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": (
                    f"QUESTION: {question}\n\n"
                    f"ANSWER: {answer}\n\n"
                    f"CONTEXT:\n{ctx_truncated}\n\n"
                    f"GROUND TRUTH (reference): {ground_truth}"
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "score": result.get("score", 0),
            "reasoning": result.get("reasoning", ""),
            "tokens": response.usage.total_tokens,
        }

    except Exception as e:
        logger.error(f"Judge failed: {e}")
        return {"score": 0, "reasoning": f"Judge error: {e}", "tokens": 0}


# Main evaluation runner 

def run_evaluation(ticker: str = "AAPL", use_judge: bool = True) -> dict:
    print(f"\n{'='*60}")
    print(f"  Evaluation: {ticker}")
    print(f"  Mode: {'Reliability + Accuracy (LLM judge)' if use_judge else 'Reliability only (rules)'}")
    print(f"{'='*60}\n")

    questions = AAPL_TEST_QUESTIONS if ticker == "AAPL" else AAPL_TEST_QUESTIONS
    results = []

    for i, (capability, question, ground_truth) in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {capability}...")

        try:
            response = ask(question, ticker, show_context=True)
            answer = response["answer"]
            context = response.get("context", "")
            validation = response.get("validation", {})

            citations = _count_citations(answer)
            has_refusal = _has_refusal(answer)
            has_numbers = _has_dollar_amounts(answer)
            guardrail_warnings = len(validation.get("warnings", []))
            sources_used = list({s["source_type"] for s in response.get("sources", [])})
            tools_called = response.get("routing", {}).get("tools_called", [])

            # Rule-based pass: has citations OR explicit refusal
            rule_passed = citations > 0 or has_refusal

            # LLM grounding score
            grounding = {"score": 0, "reasoning": "skipped", "tokens": 0}
            if use_judge:
                grounding = _judge_grounding(question, answer, context, ground_truth)

            result_entry = {
                "capability": capability,
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "citations": citations,
                "has_refusal": has_refusal,
                "has_numbers": has_numbers,
                "guardrail_warnings": guardrail_warnings,
                "sources_used": sources_used,
                "tools_called": tools_called,
                "tokens": response.get("token_usage", {}).get("total_tokens", 0),
                "rule_passed": rule_passed,
                "grounding_score": grounding["score"],
                "grounding_reasoning": grounding["reasoning"],
                "judge_tokens": grounding.get("tokens", 0),
            }
            results.append(result_entry)

            # Status line
            cite_status = f"{citations} cites" if citations > 0 else ("REFUSED" if has_refusal else "NO CITES")
            ground_status = f"ground={grounding['score']}/5" if use_judge else ""
            print(f"      → {cite_status}, {ground_status}, tools: {tools_called}, "
                  f"{guardrail_warnings} warnings, {result_entry['tokens']} tokens")

        except Exception as e:
            logger.error(f"Failed: {e}")
            results.append({
                "capability": capability,
                "question": question,
                "answer": f"ERROR: {e}",
                "citations": 0,
                "has_refusal": False,
                "guardrail_warnings": 0,
                "sources_used": [],
                "tools_called": [],
                "tokens": 0,
                "rule_passed": False,
                "grounding_score": 0,
                "grounding_reasoning": f"Error: {e}",
                "judge_tokens": 0,
            })

    # ── Scorecard ─────────────────────────────────────────────────────
    total = len(results)
    cited = sum(1 for r in results if r["citations"] > 0)
    refused = sum(1 for r in results if r.get("has_refusal", False))
    rule_passed = sum(1 for r in results if r["rule_passed"])
    total_citations = sum(r["citations"] for r in results)
    total_warnings = sum(r["guardrail_warnings"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    total_judge_tokens = sum(r.get("judge_tokens", 0) for r in results)
    with_numbers = sum(1 for r in results if r.get("has_numbers", False))

    # Source coverage
    uses_sec = sum(1 for r in results if "sec" in r.get("sources_used", []))
    uses_fmp = sum(1 for r in results if "fmp" in r.get("sources_used", []))
    uses_news = sum(1 for r in results if "news" in r.get("sources_used", []))

    # Grounding scores
    grounding_scores = [r["grounding_score"] for r in results if r["grounding_score"] > 0]
    avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0
    grounding_dist = {s: sum(1 for g in grounding_scores if g == s) for s in range(1, 6)}

    print(f"\n{'─'*60}")
    print(f"  SCORECARD: {ticker}")
    print(f"{'─'*60}\n")

    print(f"  ── Axis 1: RELIABILITY (rule-based) ──")
    print(f"  Questions passed:          {rule_passed}/{total} ({rule_passed/total*100:.0f}%)")
    print(f"    With citations:          {cited}/{total}")
    print(f"    With explicit refusal:   {refused}/{total}")
    print(f"  Total citations:           {total_citations} (avg {total_citations/total:.1f}/answer)")
    print(f"  Answers with $ amounts:    {with_numbers}/{total}")
    print(f"  Guardrail warnings:        {total_warnings}")

    if use_judge:
        print(f"\n  ── Axis 2: ACCURACY (LLM-as-judge) ──")
        print(f"  Avg grounding score:       {avg_grounding:.1f}/5.0")
        print(f"  Score distribution:        {dict(grounding_dist)}")
        high_ground = sum(1 for g in grounding_scores if g >= 4)
        print(f"  Well-grounded (≥4/5):      {high_ground}/{len(grounding_scores)}")

    print(f"\n  ── Source Coverage ──")
    print(f"    SEC filings used:        {uses_sec}/{total}")
    print(f"    FMP data used:           {uses_fmp}/{total}")
    print(f"    News used:               {uses_news}/{total}")

    print(f"\n  ── Cost ──")
    print(f"    Pipeline tokens:         {total_tokens:,}")
    print(f"    Judge tokens:            {total_judge_tokens:,}")
    print(f"    Total tokens:            {total_tokens + total_judge_tokens:,}")

    # Failures
    failures = [r for r in results if not r["rule_passed"]]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f['capability']}: no citations and no refusal")

    # Low grounding scores
    if use_judge:
        low_ground = [r for r in results if 0 < r["grounding_score"] <= 2]
        if low_ground:
            print(f"\n  LOW GROUNDING ({len(low_ground)}):")
            for lg in low_ground:
                print(f"    - {lg['capability']} (score {lg['grounding_score']}): {lg['grounding_reasoning'][:80]}")

    # Guardrail warnings
    warned = [r for r in results if r["guardrail_warnings"] > 0]
    if warned:
        print(f"\n  GUARDRAIL WARNINGS ({len(warned)} questions):")
        for w in warned:
            print(f"    - {w['capability']}: {w['guardrail_warnings']} warning(s)")

    print(f"\n{'─'*60}\n")

    # ── Save results ─────────────────────────────────────────────────
    eval_dir = project_root / "eval_results"
    eval_dir.mkdir(exist_ok=True)
    output_file = eval_dir / f"eval_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "use_judge": use_judge,
            "judge_model": EVAL_MODEL if use_judge else None,
            "questions": total,
        },
        "scorecard": {
            "reliability": {
                "questions_passed": f"{rule_passed}/{total}",
                "pass_rate": round(rule_passed / total * 100, 1),
                "total_citations": total_citations,
                "avg_citations_per_answer": round(total_citations / total, 1),
                "guardrail_warnings": total_warnings,
            },
            "accuracy": {
                "avg_grounding_score": round(avg_grounding, 2),
                "score_distribution": grounding_dist,
                "well_grounded_pct": round(high_ground / len(grounding_scores) * 100, 1) if grounding_scores else 0,
            } if use_judge else None,
            "cost": {
                "pipeline_tokens": total_tokens,
                "judge_tokens": total_judge_tokens,
                "total_tokens": total_tokens + total_judge_tokens,
            },
        },
        "results": results,
    }

    serializable = json.loads(json.dumps(output, default=str))
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"  Results saved to: {output_file}\n")

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    use_judge = "--no-judge" not in sys.argv
    run_evaluation(ticker, use_judge=use_judge)
