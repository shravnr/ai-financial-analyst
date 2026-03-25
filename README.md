# AI Financial Analyst Chatbot

CLI chatbot that answers questions about US public companies using SEC filings, financial data, and news. Every answer is grounded in retrieved data and cited to its source.

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

Ask about any company's financials in natural language. Financial data and news are fetched live per question.

## Architecture

```
User question
  -> Router classifies intent and extracts ticker (regex + gpt-5.4-nano fallback)
  -> If new company: auto-ingests SEC filings -> chunks -> ChromaDB
  -> Agentic tool loop (gpt-5.4-mini, up to 3 iterations):
       LLM picks which tools to call based on the question:
         search_sec_filings  (10-K, 10-Q, 8-K) — pre-indexed, hybrid retrieval
         get_financial_data  (selective categories) — live FMP API, LLM picks only needed data
         get_news            (relevance-filtered) — live NewsAPI, top 5 by question relevance
         get_stock_quote     (price, volume, market cap) — live FMP API
       SEC: hybrid retrieval (vector + BM25) -> relevant chunks with source IDs
       FMP: LLM selects specific categories (e.g. income_statement, metrics) -> targeted API calls
       News: fetch 15 articles -> keyword relevance ranking -> top 5 sent to LLM
       LLM sees filtered results, can call more tools or produce final answer
  -> Post-retrieval data boundary injection (actual dates from chunks, not date math)
  -> Answer with inline [N] citations
  -> Guardrails validate the answer:
       Rule-based: citations present? numbers traceable to sources?
       LLM grounding: do cited sources actually support the claims? (gpt-5.4-nano)
       If issues found: one correction pass feeds warnings back to the LLM
  -> Citation postprocessor: strips LLM-generated footnotes, rebuilds from real metadata
```

## Key Design Decisions

### Retrieval

**Agentic tool calling over fixed retrieval.** The LLM decides what to fetch based on the question. A risk question pulls SEC filings; a margin question hits FMP live; a deeper question like "how much is R&D and why" calls both to combine the numbers with management commentary. Capped at 3 tool iterations.

**Three retrieval strategies for three data types.**

- **SEC filings:** hybrid search with ~20 candidates each from vector similarity and BM25 keyword matching, merged down to 8-12 chunks depending on query complexity.
- **Financial data:** fetched live from FMP across 7 categories (income statement, balance sheet, cash flow, metrics, profile, analyst grades, analyst estimates). The LLM picks only the categories the question actually needs instead of calling all 7, each returning the last 2 fiscal years.
- **News:** fetched live (15 candidates), ranked by keyword relevance to the question, top 5 sent to the LLM.

**Retrieval-based boundaries.** Instead of guessing data scope from dates (breaks on fiscal year mismatches), the system injects actual retrieved date ranges into tool responses. The LLM sees exactly which periods it has and responds accordingly.

### Grounding and Safety

**Zero general knowledge.** The LLM is instructed to treat its training data as nonexistent. If the tools return nothing useful, it says so. This is the primary hallucination prevention, reinforced by the guardrails layer.

**Inline citations from real metadata.** Every claim maps to a numbered source. The system strips any LLM-generated footnotes and rebuilds citations from actual metadata: SEC filing URLs, FMP statement types and dates, news article links. Numbers are traced back to their source during validation.

**Two-layer guardrails with self-correction.** Rule-based checks catch missing citations and unverifiable numbers. LLM grounding catches cases where a citation exists but doesn't actually support the claim. If either layer flags issues, one correction pass feeds the warnings back to the LLM. The corrected answer is only accepted if it reduces warnings.

### Evaluation

```bash
python -m src.evaluation.run_eval
```

No external datasets. Auto-ingests tickers, questions are templatized so it works for any company. Runs a deep eval on one ticker and a smoke test on a different one, then prints a scorecard. Five signals split across two methods:

- **LLM-as-judge**: consistency, grounding, boundary refusal
- **Rule-based**: citation and number verification, smoke test

Master score is the average of LLM-judge and rule-based group scores. Latest: 9.7/10 (AMZN), 9.5/10 (AAPL) - Full eval results with per-question breakdowns are included in `eval_results/`. Every change to retrieval, prompts, or guardrails is run through this eval before being considered done. It acts as a regression gate to catch quality drops early.

### Cost

**Model tiering.** gpt-5.4-mini where accuracy matters (answers, eval). gpt-5.4-nano where it doesn't (routing, validation). Keeps costs low without sacrificing quality.

## Tradeoffs

**Free-tier API constraints.** FMP and NewsAPI hit rate limits quickly. When unavailable, the AI says so honestly rather than hallucinating. Earnings call transcripts aren't on any free API; 8-K press releases from SEC partially cover this. Paid APIs would unlock richer data and higher throughput.

**Single-company context.** Queries run against one company at a time. Cross-company comparisons would require parallel tool execution across tickers.

**Latency.** First question for a new company takes ~30s to download and index SEC filings. After that, it's fast. It's not optimized for latency yet, since accuracy was priority for this scope. A background pre-indexing job and keeping the critical path lean would eliminate the wait.

**Local storage by design.** ChromaDB and raw filings live on disk, sufficient for this scope. SEC data isn't refreshed between sessions. A hosted vector database with scheduled refresh would keep data current at scale.

## Stack

- Python 3, gpt-5.4-mini for answers and eval, gpt-5.4-nano for routing and validation, text-embedding-3-small (OpenAI) for embeddings, ChromaDB for vector storage, BM25 (rank-bm25) for keyword search, LangChain for chunking, SEC EDGAR, FMP API, NewsAPI for data sources, Rich for CLI rendering
