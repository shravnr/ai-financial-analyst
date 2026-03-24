# AI Financial Analyst Chatbot

CLI chatbot that answers questions about US public companies using SEC filings, financial data, and news. Every answer is grounded in retrieved data and cited to its source.

## Stack

- Python 3.10+
- gpt-5.4-mini for answers and eval, gpt-5.4-nano for routing and validation
- text-embedding-3-small (OpenAI) for embeddings
- ChromaDB for vector storage, BM25 (rank-bm25) for keyword search
- LangChain for chunking
- SEC EDGAR, FMP API, NewsAPI for data
- Rich for CLI rendering

## Quick Start

1. Clone the repo and `cd ai-financial-analyst-chatbot`
2. Create a virtualenv: `python3 -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your API keys
5. Run: `python main.py`

`.env` needs:

- `OPENAI_API_KEY` (platform.openai.com)
- `FMP_API_KEY` (site.financialmodelingprep.com, free tier)
- `NEWS_API_KEY` (newsapi.org, free tier)
- `SEC_EDGAR_USER_AGENT` (your name + email, SEC requires it)

Ask about any company's financials in natural language. First query for a new company takes ~30s to pull and index data, then it's quick.

## Architecture

```
User question
  -> Router classifies intent and extracts ticker (regex + gpt-5.4-nano fallback)
  -> If new company: auto-ingests SEC + FMP + News -> chunks -> ChromaDB
  -> Agentic tool loop (gpt-5.4-mini, up to 3 iterations):
       LLM picks which tools to call based on the question:
         search_sec_filings   (10-K, 10-Q, 8-K)
         search_financial_data (income stmt, balance sheet, cash flow, metrics, estimates)
         search_news          (last ~30 days)
       Each tool: hybrid retrieval (vector + BM25 via RRF) -> formatted chunks with source IDs
       LLM sees results, can call more tools or produce final answer
  -> Post-retrieval data boundary injection (actual dates from chunks, not date math)
  -> Answer with inline [N] citations
  -> Guardrails validate the answer:
       Rule-based: citations present? numbers traceable to sources?
       LLM grounding: do cited sources actually support the claims? (gpt-5.4-nano)
  -> Citation postprocessor: strips LLM-generated footnotes, rebuilds from real metadata
```

## Key Design Decisions

**Agentic tool calling over fixed retrieval.** The LLM decides what to fetch based on the question. A risk question pulls SEC filings; a revenue question pulls financial statements. Capped at 3 iterations to bound cost.

**Hybrid search.** Semantic embeddings catch meaning; BM25 catches exact terms like ticker symbols and dollar amounts. Merged via Reciprocal Rank Fusion. Neither alone is sufficient for financial data.

**Model tiering.** gpt-5.4-mini where accuracy matters (answers, eval). gpt-5.4-nano where it doesn't (routing, validation). Keeps costs low without sacrificing quality.

**Retrieval-based boundaries.** Instead of guessing data scope from dates (breaks on fiscal year mismatches), the system injects actual retrieved date ranges into tool responses. The LLM sees exactly which periods it has and responds accordingly.

**Two-layer guardrails.** Rule-based checks catch missing citations and unverifiable numbers. LLM grounding catches cases where a citation exists but doesn't actually support the claim.

## Evaluation

```bash
python -m src.evaluation.run_eval
```

No external datasets. Auto-ingests tickers, questions are templatized so it works for any company. Change `DEEP_EVAL_TICKER` in `run_eval.py` to pick a different one.

Runs a deep eval (14 questions) on one ticker and a smoke test on a different one, then prints a scorecard. Five signals split across two methods:

- **LLM-as-judge** (gpt-5.4-mini): consistency (4 pairs), grounding (10 questions), boundary refusal (6 cases)
- **Rule-based**: citation verification, number traceability, smoke test (3 questions on second ticker)

Master score = average of LLM-judge and rule-based group scores.

## Project Structure

```
main.py                              CLI entry point, intent router, conversation loop
src/
  config.py                          API keys, model names, paths, ingestion defaults
  ingestion/
    orchestrator.py                  Coordinates all data fetching for a ticker
    sec_edgar.py                     SEC EDGAR full-text filing downloader (10-K, 10-Q, 8-K)
    fmp.py                           FMP API client (financials, metrics, ratios, estimates)
    news.py                          NewsAPI client
  processing/
    pipeline.py                      Orchestrates chunking and indexing
    chunker.py                       SEC filing chunker with section detection
    structured_formatter.py          Converts FMP JSON to readable text for embedding
    vector_store.py                  ChromaDB + BM25 hybrid search with RRF merging
  rag/
    chain.py                         Agentic tool-calling loop, citation postprocessor
    query_router.py                  Regex + LLM hybrid query classifier
  guardrails/
    validator.py                     Citation check, number verification, LLM grounding
  evaluation/
    run_eval.py                      Eval runner and scorecard
    test_questions.py                Templatized question bank
```

## Tradeoffs

**Data coverage.** FMP free tier provides annual financials only (no quarterly breakdowns; SEC 10-Q partially covers this). NewsAPI free tier limits to ~30 days. No earnings call transcripts or real-time market data. SEC filings go back ~2 years (configurable). All free-tier constraints; upgrading the APIs would directly expand coverage.

**No query decomposition.** Complex multi-part questions ("compare revenue, margins, and debt") run as a single query per tool call. Breaking these into sub-queries and merging results would improve recall, but added latency and complexity weren't justified for this scope.

**Single-company context.** The agentic loop searches one ticker at a time, so cross-company comparisons ("AAPL margin vs MSFT") aren't supported in a single query. Supporting this would require parallel tool execution across tickers.

**~30s cold start per new company.** First query ingests SEC filings, FMP data, and news, then chunks and embeds everything. Subsequent queries are fast. Accuracy was prioritized over latency for this scope; a background pre-indexing job would eliminate the wait.
