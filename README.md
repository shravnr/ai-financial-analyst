# AI Financial Analyst Chatbot

An AI-powered financial analyst that answers questions about any US public company using SEC filings, financial data, and recent news — with citations on every claim.

## Quick Start

**Requirements:** Python 3.10+, API keys for OpenAI, FMP, and NewsAPI.

```bash
# Clone and setup
cd ai-financial-analyst-chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env with your keys (OPENAI_API_KEY, FMP_API_KEY, NEWS_API_KEY)

# Run the chatbot
python main.py
```

Just ask a question like *"What is Apple's revenue?"* — the system auto-detects the ticker, ingests data (~30s for a new company), and answers with citations. Subsequent questions about the same company are instant.

## How It Works

```
"What is Apple's revenue?"
        │
        ▼
┌──────────────────┐
│  Input Router     │  ← gpt-4o-mini detects company, resolves ticker
│  Auto-ingest if   │    SEC + FMP + News → ChromaDB (one-time, ~30s)
│  new company       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────┐
│  GPT-4o + Tools   │────▶│  ChromaDB (500+ chunks/company) │
│                   │     │  ├─ SEC: 10-K, 10-Q, 8-K       │
│  Tools:           │     │  ├─ FMP: statements, metrics    │
│  • search_sec     │     │  └─ News: recent articles       │
│  • search_financial│    └─────────────────────────────────┘
│  • search_news    │
└────────┬─────────┘
         │  LLM decides which tools to call (agentic loop, max 3 iterations)
         ▼
┌──────────────────┐
│  Guardrails       │  ← Rule-based (citation, number verify, sufficiency)
│                   │    + LLM grounding check (gpt-4o-mini)
└────────┬─────────┘
         ▼
   Cited answer with [Source N] references
```

**Key design choices:**

- **Agentic tool calling** — GPT-4o reasons about what data it needs rather than following a fixed retrieval pipeline. A risk question calls SEC filings; a valuation question calls FMP data; complex questions call multiple tools.
- **Hybrid query router** — Regex fast path handles ~80% of queries (free, instant). LLM fallback (gpt-4o-mini) handles the rest.
- **Layered guardrails** — Rule-based checks catch structural issues (missing citations, unverifiable numbers). LLM grounding catches semantic hallucination ("cited Source 3, but Source 3 doesn't say that").
- **Model tiering** — gpt-4o for answer generation and eval judging. gpt-4o-mini for routing, grounding validation, and ticker detection. Not every task needs the most capable model.

## Data Sources

| Source        | What It Provides                                             | Free Tier                |
| ------------- | ------------------------------------------------------------ | ------------------------ |
| **SEC EDGAR** | 10-K, 10-Q, 8-K filings                                     | Unlimited (no key needed)|
| **FMP**       | Financial statements, key metrics, ratios, analyst estimates | 250 calls/day            |
| **NewsAPI**   | Recent news articles                                         | 100 calls/day            |

## Project Structure

```
├── main.py                          # Interactive CLI chatbot
├── src/
│   ├── config.py                    # Central config (API keys, models, paths)
│   ├── ingestion/                   # Data collection from 3 sources
│   │   ├── sec_edgar.py             # SEC EDGAR API client
│   │   ├── fmp.py                   # FMP API client
│   │   ├── news.py                  # NewsAPI client
│   │   └── orchestrator.py          # Coordinates all sources for a ticker
│   ├── processing/                  # Chunking + embedding into ChromaDB
│   │   ├── chunker.py              # Text splitting + 10-K section detection
│   │   ├── structured_formatter.py  # FMP JSON → readable text
│   │   ├── vector_store.py         # ChromaDB operations
│   │   └── pipeline.py             # End-to-end processing orchestrator
│   ├── rag/                         # Retrieval + generation
│   │   ├── query_router.py         # Hybrid router (regex → LLM fallback)
│   │   └── chain.py                # Agentic tool-calling loop (GPT-4o)
│   ├── guardrails/                  # Post-generation validation
│   │   └── validator.py            # Rule checks + LLM grounding
│   └── evaluation/                  # Self-contained eval suite
│       ├── test_questions.py        # Templatized questions (any ticker)
│       └── run_eval.py             # 5-signal eval with unified scorecard
├── data/                            # Auto-generated on first use (gitignored)
├── .env.example                     # API key template
└── requirements.txt
```

## Evaluation

The eval suite runs end-to-end with no external dependencies or pre-built datasets. It auto-ingests tickers as needed.

```bash
python -m src.evaluation.run_eval
```

This runs a **deep eval** (14 questions across consistency, grounding, and verification) on one ticker, plus a **smoke test** (3 questions + 3 boundary refusals) on a separate randomly-selected ticker. Both tickers are configurable in `run_eval.py`.

### Scorecard (5 signals, 2 axes)

**LLM-as-judge** — evaluated by gpt-4o:
- **Consistency** — Same question asked two ways; LLM judge scores agreement (4 pairs)
- **Grounding** — Are factual claims supported by the retrieved context? (10 questions)
- **Boundary** — Does the system refuse gracefully for private/unknown/speculative companies? (3 cases)

**Rule-based** — deterministic checks:
- **Verification** — Citation presence + number traceability against source context
- **Smoke** — Breadth test across a different ticker (citations present, sources retrieved)

```
Master Score = avg(LLM-judge avg, Rule-based avg)
```

Eval is ticker-agnostic — questions use `{company}` templates resolved at runtime. To eval a different company, change `DEEP_EVAL_TICKER` in `run_eval.py`.

### Sample Results

| Ticker | Master Score | Verdict |
| ------ | ------------ | ------- |
| AAPL   | 9.2 / 10     | PASS    |
| AMZN   | 9.5 / 10     | PASS    |

> Consistency scores may vary +/-1 pair between runs due to LLM non-determinism. This is expected.

## Constraints Enforced

1. **No uncited claims** — Every factual statement requires a `[Source N]` citation. Guardrails flag violations.
2. **No LLM recall** — System prompt forbids using training data for financial facts. Only retrieved data is used.
3. **No hallucinated numbers** — Dollar amounts are checked against source context (rule-based + LLM grounding).
4. **Explicit refusal** — When context is insufficient, the system says so rather than guessing.

## Known Limitations

- **FMP free tier** — No quarterly statements or earnings transcripts. Mitigated by SEC 10-Q filings and 8-K press releases.
- **NewsAPI free tier** — Limited date filtering. Articles may not be the most recent.
- **Number verification false positives** — Unit conversions (e.g., thousands → billions) may trigger guardrail warnings on valid numbers.
- **First-query latency** — ~30s to ingest a new company. Subsequent queries are instant.

## API Keys

| Service | Sign Up                         | Free Tier     |
| ------- | ------------------------------- | ------------- |
| OpenAI  | platform.openai.com             | Pay-per-use   |
| FMP     | site.financialmodelingprep.com  | 250 calls/day |
| NewsAPI | newsapi.org                     | 100 calls/day |
