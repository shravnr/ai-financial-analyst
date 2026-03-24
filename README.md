# AI Financial Analyst Chatbot

CLI chatbot that answers questions about US public companies using SEC filings, financial data, and news. Every answer is cited.

## Quick Start

Python 3.10+.

```bash
cd ai-financial-analyst-chatbot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys
python main.py
```

`.env` needs:

- `OPENAI_API_KEY` (platform.openai.com)
- `FMP_API_KEY` (site.financialmodelingprep.com, free tier)
- `NEWS_API_KEY` (newsapi.org, free tier)
- `SEC_EDGAR_USER_AGENT` (your name + email, SEC requires it)

Mention a company and it takes care of the rest. First time for a company takes ~30s to pull and index data, then it's instant.

## Stack

- GPT-5.4-mini for answers, GPT-5.4-nano for routing and validation
- SEC EDGAR, FMP API, NewsAPI for data
- ChromaDB + OpenAI embeddings for retrieval
- LangChain for chunking, OpenAI function calling for tool use

## Architecture

```
Question
  -> Router identifies company and ticker (gpt-5.4-nano)
  -> Auto-ingests if company is new (SEC + FMP + News -> ChromaDB)
  -> gpt-5.4-mini decides which tools to call (up to 3 rounds):
       search_sec_filings, search_financial_data, search_news
  -> Guardrails check the answer:
       Rule-based: citations present? numbers traceable to sources?
       LLM-based: do the cited sources actually support the claims?
  -> Answer with [Source N] citations
```

It's not a fixed retrieval pipeline. The LLM picks which sources to query based on the question, so a risk question pulls SEC filings while a revenue question pulls financial statements.

## Design Choices

**Agentic tool calling.** The LLM decides what data to fetch rather than always querying everything. Costs more tokens but gives better answers. Capped at 3 iterations.

**Model tiering.** gpt-5.4-mini where accuracy matters (answers, eval). gpt-5.4-nano where it doesn't (routing, validation). Keeps costs down without hurting quality.

**Hybrid router.** Regex handles most queries for free. LLM fallback catches the rest.

**Two-layer guardrails.** Rules catch missing citations and bad numbers. LLM grounding catches cases where a citation exists but doesn't actually support the claim.

## Evaluation

```bash
python -m src.evaluation.run_eval
```

No external datasets needed. Auto-ingests tickers, questions are templatized so it works for any company. Change `DEEP_EVAL_TICKER` in `run_eval.py` to pick a different one.

Runs a deep eval (14 questions) on one ticker and a smoke test on a different one, then prints a scorecard. Five signals split across two methods:

- LLM-as-judge (gpt-5.4-mini): consistency, grounding, boundary refusal
- Rule-based: citation/number verification, smoke test

Master score = average of both groups. Scored 9.5 on AAPL and 9.3 on AMZN.

## Limitations

- FMP free tier skips quarterly data and transcripts (SEC 10-Q and 8-K help cover this)
- NewsAPI free tier has limited date filtering
- Number guardrail can false-positive on unit conversions
- ~30s cold start per new company
