# AI Financial Analyst Chatbot

A conversational financial analyst that can answer questions about any US public company. It pulls data from SEC filings, financial APIs, and news sources, then generates cited answers grounded in that data.

Ask *"What are Apple's biggest risks?"* and it fetches the 10-K, finds the risk factors section, and gives you a sourced answer with `[Source N]` citations. Ask about revenue and it pulls financial statements instead. It figures out what data it needs.

## Quick Start

Python 3.10+ required.

```bash
cd ai-financial-analyst-chatbot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in the values below
python main.py
```

Your `.env` needs:
- `OPENAI_API_KEY` from platform.openai.com (pay-per-use)
- `FMP_API_KEY` from site.financialmodelingprep.com (free, 250 calls/day)
- `NEWS_API_KEY` from newsapi.org (free, 100 calls/day)
- `SEC_EDGAR_USER_AGENT` your name and email (SEC requires this, no key needed)

First query for a new company takes ~30s while it ingests data. After that it's instant.

## Resources

- **Models**: GPT-4o (answer generation, eval judging), GPT-4o-mini (routing, grounding checks, ticker detection)
- **Data**: SEC EDGAR (10-K, 10-Q, 8-K filings), FMP (financial statements, metrics, ratios, estimates), NewsAPI (recent articles)
- **Vector store**: ChromaDB with OpenAI embeddings (text-embedding-3-small)
- **Framework**: LangChain (chunking), OpenAI function calling (agentic tool use)

## Architecture

```
User question
  -> Input router (gpt-4o-mini): detects company, resolves ticker, auto-ingests if new
  -> GPT-4o agentic loop (up to 3 tool calls):
       search_sec_filings       pulls from 10-K, 10-Q, 8-K chunks in ChromaDB
       search_financial_data    pulls from FMP statements, metrics, ratios
       search_news              pulls from recent news articles
  -> Guardrails:
       Rule-based               citation check, number verification, sufficiency
       LLM grounding            gpt-4o-mini checks if citations actually support claims
  -> Cited answer with [Source N] references
```

## Key Design Choices

**Agentic tool calling over static RAG.** The LLM decides which data sources to query based on the question. A risk question hits SEC filings. A valuation question hits FMP. A broad question hits multiple sources. This is the core difference from a fixed retrieve-then-generate pipeline. Tradeoff: 2-3x more tokens per question, capped at 3 tool iterations.

**Two-tier model strategy.** Not every task needs GPT-4o. Routing, ticker detection, and grounding validation are classification tasks that GPT-4o-mini handles well at a fraction of the cost. GPT-4o is reserved for answer generation and eval judging where accuracy matters most.

**Hybrid query router.** Regex patterns handle ~80% of financial queries (free, instant, deterministic). The remaining edge cases fall through to GPT-4o-mini. Neither approach alone is sufficient.

**Layered guardrails.** Rule-based checks catch structural problems (missing citations, numbers not traceable to context). But rules can't catch "cited Source 3, but Source 3 doesn't actually say that." The LLM grounding layer catches that kind of semantic hallucination.

## Evaluation

```bash
python -m src.evaluation.run_eval
```

Self-contained, no external datasets. Auto-ingests tickers as needed. Questions are templatized with a `{company}` placeholder so the eval works for any ticker. Change `DEEP_EVAL_TICKER` in `run_eval.py` to point at a different company.

It runs a deep eval (14 questions) on one ticker and a smoke test (3 questions + boundary refusals) on a separate randomly-picked ticker, then prints a unified scorecard.

Five signals across two methods:

LLM-as-judge (GPT-4o) scores consistency (same question asked two ways), grounding (are claims supported by context?), and boundary behavior (graceful refusal for unknown/private companies). Rule-based checks score verification (citation + number traceability) and smoke (breadth across a different ticker). Master score averages both groups.

Tested on AAPL (9.2/10) and AMZN (9.5/10). Consistency can vary +/-1 pair between runs due to LLM non-determinism.

## Limitations

- FMP free tier doesn't include quarterly statements or earnings transcripts. SEC 10-Q and 8-K filings partially fill this gap.
- NewsAPI free tier has limited date filtering, so articles may not always be the most recent.
- The number verification guardrail can false-positive when the LLM converts units (e.g. SEC reports in thousands, answer in billions).
- First query per company takes ~30s for data ingestion.
