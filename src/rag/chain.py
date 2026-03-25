import json
import logging
import re
from datetime import datetime

from openai import OpenAI

import requests

from src.config import OPENAI_API_KEY, LLM_MODEL, NEWS_API_KEY, NEWS_API_BASE_URL
from src.ingestion.fmp import _fmp_get, FMP_RATE_LIMITED, FMP_PAYMENT_REQUIRED
from src.processing.structured_formatter import (
    _format_profile, _format_income_statement, _format_balance_sheet,
    _format_cash_flow, _format_key_metrics, _format_grades, _format_analyst_estimates,
)
from src.processing.vector_store import query as vector_query
from src.rag.query_router import route_query
from src.guardrails.validator import validate_response

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

MAX_TOOL_ITERATIONS = 3

# Tool definitions for function calling

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_sec_filings",
            "description": "Search SEC filings (10-K annual reports, 10-Q quarterly reports, 8-K earnings press releases) for a company. Best for: risks, strategy, business segments, legal disclosures, management discussion, qualitative analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what information to find in SEC filings",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default 8)",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": (
                "Fetch live financial data for a company from FMP API. "
                "Select ONLY the categories relevant to the question — do NOT request all.\n\n"
                "Categories:\n"
                "- profile: company overview, sector, CEO, market cap, description\n"
                "- income_statement: revenue, operating income, net income, EPS, margins\n"
                "- balance_sheet: assets, liabilities, equity, debt, cash positions\n"
                "- cash_flow: operating/investing/financing cash flows, free cash flow, capex\n"
                "- metrics: valuation ratios (P/E, EV/EBITDA, P/S, P/B), returns (ROE, ROIC), debt ratios\n"
                "- analyst_grades: recent analyst rating changes (buy/sell/hold) from major firms\n"
                "- analyst_estimates: consensus revenue, EPS, EBITDA forecasts"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "profile", "income_statement", "balance_sheet",
                                "cash_flow", "metrics", "analyst_grades", "analyst_estimates",
                            ],
                        },
                        "description": "Which data categories to fetch. Pick only what the question needs.",
                    },
                },
                "required": ["categories"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Fetch the latest news articles about a company from the last 30 days. Hits the NewsAPI directly for current articles. Best for: latest developments, current events, market sentiment, recent announcements.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Get the current live stock quote for a company — price, change, volume, market cap. Use this for any question about current/today's stock price, market cap, or trading activity. This hits a live API, not pre-indexed data.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

_TOOL_SOURCE_MAP = {
    "search_sec_filings": ["sec"],
    "get_financial_data": None,
    "get_news": None,
    "get_stock_quote": None,
}

SYSTEM_PROMPT = """You are a professional, straightforward financial analyst. Direct, specific, and concise.

## RULES

1. Call tools to retrieve relevant data. Use multiple tools when the question spans different data types. For get_financial_data, request ONLY the categories the question needs. **If the question mentions a specific segment, division, or product line (e.g., AWS, Google Cloud, iPhone, Azure), you MUST call search_sec_filings** — segment data is only in SEC filings, not in consolidated financial data.

   **Combine structured + unstructured data.** When a question involves numbers AND context (e.g., "how does capex compare to FCF" or "what's driving margin expansion"), call BOTH get_financial_data for the numbers AND search_sec_filings for management commentary. Numbers alone are incomplete; context alone lacks precision.

   **If news results are irrelevant or empty**, still answer the topical part of the question using search_sec_filings. For example, if asked about "AI strategy" and news returns unrelated articles, search SEC filings for the topic — 10-K/10-Q filings contain extensive strategy and initiative discussion.

   Tool selection examples:
   - "What's revenue and margins?" → get_financial_data(categories=["income_statement"])
   - "What's the P/E ratio and valuation?" → get_financial_data(categories=["metrics"])
   - "How much debt does it have?" → get_financial_data(categories=["balance_sheet"]) + search_sec_filings(query="debt maturity schedule obligations")
   - "What are the biggest risks?" → search_sec_filings(query="risk factors")
   - "How much revenue does AWS generate?" → search_sec_filings(query="AWS revenue segment breakdown")
   - "What's driving capex growth?" → get_financial_data(categories=["cash_flow"]) + search_sec_filings(query="capital expenditure investment strategy")
   - "Should I buy this stock?" → get_financial_data(categories=["income_statement", "metrics", "analyst_grades"]) + search_sec_filings(query="risk factors")
   - "What's the latest news on AI strategy?" → get_news + search_sec_filings(query="AI strategy initiatives")
   - "What's the current stock price?" → get_stock_quote

2. Answer using ONLY the data your tools returned. Every claim — numbers, facts, events, qualitative statements — must have an inline [N] citation matching the exact source it came from.
3. Derived metrics (margins, growth rates, ratios, differences) computed from returned data are encouraged — that is analysis, not fabrication.
4. If tools return data relevant to the question, USE it to build the best answer you can — even if the match isn't exact. For example, an earnings summary can be built from 8-K filings and income statement data.
5. Never fabricate data or cite a source for data it does not contain. If the tools returned nothing useful, say so plainly in one sentence with NO citations — do not cite sources you searched but found nothing in.

## CITATIONS

Place [N] inline immediately after the SPECIFIC claim it supports — not at the end of the sentence. N = the Source number shown in tool results (e.g., [Source 3] → cite as [3]).

Critical rules:
- Attach each [N] to the exact number or fact it supports. If $391B comes from Source 1 and $416B comes from Source 2, write "$391B[1]" and "$416B[2]" — never "$391B[1][2]".
- Only cite a source for data that source actually contains. Never pile multiple citations unless each source independently supports the claim.
- Do NOT cite statements about data limitations or what you don't have. "I couldn't find X" or "the filings don't state Y" should have ZERO citations. If a value is N/A, null, or missing, do NOT cite the source — just say "not available" with no citation.
- Do NOT renumber sources. Do NOT write a footnotes or references section — the system generates that automatically.

Examples:
  GOOD: Revenue grew 12% to $716.9B[1], up from $638.0B[2]. Operating margin expanded to 11.2%[1].
  GOOD: The P/E ratio is not available in the current data.
  BAD:  The P/E ratio is N/A[2][3].  ← never cite sources for missing data

## DATA COVERAGE

Your tools access data from different sources:
- **SEC filings** (pre-indexed): 10-K (annual), 10-Q (quarterly), 8-K filings from the last ~2 years. Searched via semantic search. Contains segment breakdowns (e.g., AWS, Google Cloud, product lines), risk factors, management discussion, qualitative analysis, and detailed operating data not available in consolidated financials.
- **Financial data** (live API): Select only the categories you need — profile, income_statement, balance_sheet, cash_flow, metrics, analyst_grades, analyst_estimates. Fetched live from FMP — always current. NOTE: This is company-wide consolidated data only — it does NOT have segment or division breakdowns. For segment data, use search_sec_filings.
- **News** (live API): Articles from the last ~30 days. Fetched live from NewsAPI — always current.
- **Stock quote** (live API): Real-time price, volume, market cap.

If a question falls outside this range, say so directly instead of searching.

## BOUNDARIES

- **Never give personal investment advice.** Do not say "buy", "sell", "hold", or recommend any action on a stock. Do not give directional opinions like "not attractive", "looks overvalued", or "I would not buy". If asked "should I buy X?", present the relevant data (valuation, financials, analyst ratings, risks) objectively and let the user draw their own conclusion.
- **Never predict future stock prices.** You can share analyst estimates and historical trends, but do not forecast prices.
- **Disclaimer rule:** ONLY append "This is data, not investment advice — consult a financial advisor for personal recommendations." when the user explicitly asks whether to buy, sell, hold, or invest. Do NOT add it to factual questions about revenue, risks, strategy, debt, margins, or any other data question — even if the topic touches on valuation or analyst ratings.

**CRITICAL**: You have ZERO general knowledge. Every single number, fact, date, percentage, event, risk factor, product name, and claim in your answer MUST come from the tool results returned to you. If a piece of information is not explicitly present in the tool results, do NOT include it — even if you "know" it from training. Treat your training data as nonexistent. If the tool results are thin, give a shorter answer — never pad with outside knowledge.

## STYLE

- Lead with the answer. Be direct and specific ("$108.8B in FY 2024", not "approximately $109B").
- When comparing periods, always use the most recent data available and work backwards.
- Analyze trends, comparisons, and implications across cited data points.
- Ensure brevity as much as you can. No caveats, no padding, no hedging.
- Don't offer follow-up suggestions unless the user asks.
- Never mention tools, retrieval, datasets, or technical internals."""


def _format_tool_results(
    chunks: list[dict],
    source_counter: int,
    seen_sources: dict,
    source_metadata: dict,
) -> tuple[str, int]:
    parts = []
    for chunk in chunks:
        meta = chunk["metadata"]
        ticker_str = meta.get("ticker", "")
        doc_type = meta.get("document_type", "unknown").replace("_", " ").title()
        date = meta.get("date", "")
        section = meta.get("section", "")

        section_key = section.split(">")[0].strip() if section else ""
        key = f"{ticker_str}_{doc_type}_{date}_{section_key}"
        if key in seen_sources:
            num = seen_sources[key]
        else:
            source_counter += 1
            num = source_counter
            seen_sources[key] = num
            source_metadata[num] = {
                "source_type": meta.get("source_type", ""),
                "document_type": meta.get("document_type", ""),
                "ticker": ticker_str,
                "company_name": meta.get("company_name", ""),
                "date": date,
                "section": section,
                "file_path": meta.get("file_path", ""),
                "edgar_url": meta.get("edgar_url", ""),
                "url": meta.get("url", ""),
            }

        source_type = meta.get("source_type", "").upper()
        header = f"{source_type} — {ticker_str} {doc_type}" if source_type else f"{ticker_str} {doc_type}"
        if date:
            header += f" ({date})"
        if section:
            header += f", {section}"

        parts.append(f"[Source {num}] {header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts), source_counter


def _fetch_live_quote(ticker: str) -> str:
    data = _fmp_get("quote", {"symbol": ticker})

    if data == FMP_RATE_LIMITED:
        return f"Stock quote temporarily unavailable for {ticker} (API rate limit). Try again in a few minutes."

    # If quote endpoint is paywalled, fall back to profile (has price, marketCap, volume)
    if data == FMP_PAYMENT_REQUIRED:
        data = _fmp_get("profile", {"symbol": ticker})

    if not data or (isinstance(data, list) and len(data) == 0):
        return "No live quote available for this ticker."

    q = data[0] if isinstance(data, list) else data
    parts = [f"[Source: LIVE] {ticker} — Real-time Quote"]
    fields = [
        ("Price", "price"), ("Change", "change"),
        ("Change %", "changesPercentage"), ("Day Low", "dayLow"),
        ("Day High", "dayHigh"), ("Year Low", "yearLow"),
        ("Year High", "yearHigh"), ("Market Cap", "marketCap"),
        ("Volume", "volume"), ("Avg Volume", "avgVolume"),
        ("Open", "open"), ("Previous Close", "previousClose"),
        ("EPS", "eps"), ("PE", "pe"), ("Name", "name"),
        ("Exchange", "exchange"),
        # Profile-specific fields (used when falling back)
        ("52-Week Range", "range"), ("Beta", "beta"),
    ]
    for label, key in fields:
        if key in q and q[key] is not None:
            val = q[key]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if key in ("marketCap", "volume", "avgVolume"):
                    val = f"{val:,.0f}"
                elif isinstance(val, float):
                    val = f"{val:,.2f}"
            parts.append(f"  {label}: {val}")

    timestamp = q.get("timestamp")
    if timestamp:
        try:
            dt = datetime.fromtimestamp(int(timestamp))
            parts.append(f"  As of: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except (ValueError, TypeError):
            parts.append(f"  As of: {timestamp}")

    return "\n".join(parts)


def _fetch_live_financial_data(ticker: str, categories: list[str], source_counter: int, seen_sources: dict, source_metadata: dict) -> tuple[str, int]:
    all_endpoints = {
        "profile": ("profile", {"symbol": ticker}),
        "income_statement": ("income-statement", {"symbol": ticker, "period": "annual", "limit": 2}),
        "balance_sheet": ("balance-sheet-statement", {"symbol": ticker, "period": "annual", "limit": 2}),
        "cash_flow": ("cash-flow-statement", {"symbol": ticker, "period": "annual", "limit": 2}),
        "metrics": ("key-metrics", {"symbol": ticker, "period": "annual", "limit": 2}),
        "analyst_grades": ("grades", {"symbol": ticker, "limit": 20}),
        "analyst_estimates": ("analyst-estimates", {"symbol": ticker, "period": "annual", "limit": 4}),
    }

    # Only fetch what the LLM asked for
    endpoints = {k: v for k, v in all_endpoints.items() if k in categories} if categories else all_endpoints
    logger.info(f"FMP categories requested: {categories} → fetching {list(endpoints.keys())}")

    formatters = {
        "profile": lambda d: _format_profile(d, ticker),
        "income_statement": lambda d: _format_income_statement(d, ticker, "annual"),
        "balance_sheet": lambda d: _format_balance_sheet(d, ticker, "annual"),
        "cash_flow": lambda d: _format_cash_flow(d, ticker, "annual"),
        "metrics": lambda d: _format_key_metrics(d, ticker),
        "analyst_grades": lambda d: _format_grades(d, ticker),
        "analyst_estimates": lambda d: _format_analyst_estimates(d, ticker),
    }

    parts = []
    unavailable = 0
    for name, (endpoint, params) in endpoints.items():
        data = _fmp_get(endpoint, params)
        if data in (FMP_RATE_LIMITED, FMP_PAYMENT_REQUIRED):
            unavailable += 1
            continue
        if not data:
            continue

        formatter = formatters.get(name)
        if not formatter:
            continue

        docs = formatter(data)
        for doc in docs:
            source_counter += 1
            meta = doc["metadata"]
            source_metadata[source_counter] = {
                "source_type": "fmp",
                "document_type": meta.get("document_type", name),
                "ticker": ticker,
                "company_name": meta.get("company_name", ""),
                "date": meta.get("date", ""),
                "section": meta.get("section", ""),
                "file_path": "",
                "edgar_url": "",
                "url": "",
            }
            key = f"{ticker}_fmp_{name}_{meta.get('date', '')}"
            seen_sources[key] = source_counter
            parts.append(f"[Source {source_counter}] FMP — {ticker} {meta.get('section', name)}\n{doc['text']}")

    if not parts:
        if unavailable > 0:
            return (
                f"Financial data temporarily unavailable for {ticker} (API limit). "
                f"Some data endpoints are restricted on the current plan or rate-limited. "
                f"You can still use search_sec_filings to answer from SEC filings."
            ), source_counter
        return f"No financial data available for {ticker}.", source_counter

    return "\n\n---\n\n".join(parts), source_counter


_NEWS_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "why", "when",
    "where", "who", "which", "do", "does", "did", "has", "have", "had", "will",
    "would", "could", "should", "can", "may", "might", "about", "for", "of", "in",
    "on", "at", "to", "from", "with", "and", "or", "but", "not", "its", "it",
    "this", "that", "their", "they", "them", "be", "been", "being", "i", "me",
    "my", "we", "our", "you", "your", "tell", "show", "give", "company", "stock",
    "latest", "recent", "new", "news", "current", "today",
})


def _news_relevance_score(article: dict, question_terms: set[str]) -> int:
    """Score an article by keyword overlap with the user's question."""
    text = " ".join([
        article.get("title", ""),
        article.get("description", "") or "",
        article.get("content", "") or "",
    ]).lower()
    return sum(1 for term in question_terms if term in text)


def _fetch_live_news(ticker: str, question: str, source_counter: int, seen_sources: dict, source_metadata: dict) -> tuple[str, int]:
    # Get company name from FMP profile (tolerate rate limit — fall back to ticker)
    profile = _fmp_get("profile", {"symbol": ticker})
    company_name = ""
    if profile and profile not in (FMP_RATE_LIMITED, FMP_PAYMENT_REQUIRED) and isinstance(profile, list) and len(profile) > 0:
        company_name = profile[0].get("companyName", "")

    query = company_name or ticker
    # Iteratively strip corporate suffixes (handles chained ones like "Co., Ltd.")
    prev = None
    while query != prev:
        prev = query
        query = re.sub(
            r'[,.]?\s*(Inc\.?|Corp\.?|Corporation|Incorporated|Limited|Ltd\.?'
            r'|Co\.?|plc|PLC|S\.?A\.?|N\.?V\.?|SE|AG|& Co\.?|Group|Holdings?|A/S)\s*$',
            '', query
        ).strip()
    query = query or ticker

    try:
        resp = requests.get(
            f"{NEWS_API_BASE_URL}/everything",
            params={
                "q": f'"{query}"',
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 15,
                "apiKey": NEWS_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"NewsAPI request failed for {ticker}: {e}")
        return (
            f"News temporarily unavailable for {ticker} (API error). Try again in a few minutes."
        ), source_counter

    articles = data.get("articles", [])
    if not articles:
        return f"No recent news articles found for {query}.", source_counter

    # Relevance filtering — rank articles by keyword overlap with the question, keep top 5
    question_terms = {
        w for w in re.split(r'\W+', question.lower())
        if w and w not in _NEWS_STOP_WORDS and len(w) > 2
    }
    if question_terms:
        scored = [(a, _news_relevance_score(a, question_terms)) for a in articles]
        scored.sort(key=lambda x: x[1], reverse=True)
        articles = [a for a, _ in scored[:5]]
        logger.info(f"News: filtered {len(data.get('articles', []))} articles → top {len(articles)} by relevance to: {question_terms}")

    parts = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""
        source_name = article.get("source", {}).get("name", "")
        date = (article.get("publishedAt") or "")[:10]
        url = article.get("url", "")

        # Strip NewsAPI truncation
        if content.endswith("]"):
            content = re.sub(r"\s*\[\+\d+ chars\]$", "", content)

        text = title
        if description:
            text += f"\n\n{description}"
        if content and content != description:
            text += f"\n\n{content}"

        source_counter += 1
        source_metadata[source_counter] = {
            "source_type": "news",
            "document_type": "news_article",
            "ticker": ticker,
            "company_name": company_name,
            "date": date,
            "section": f"News - {source_name}",
            "file_path": "",
            "edgar_url": "",
            "url": url,
        }
        key = f"{ticker}_news_{date}_{title[:30]}"
        seen_sources[key] = source_counter
        header = f"NEWS — {ticker} \"{title}\" ({date})"
        if source_name:
            header += f", {source_name}"
        parts.append(f"[Source {source_counter}] {header}\n{text}")

    return "\n\n---\n\n".join(parts), source_counter


def _execute_tool(
    tool_name: str,
    args: dict,
    ticker: str,
    question: str,
    source_counter: int,
    seen_sources: dict,
    source_metadata: dict,
) -> tuple[str, list[dict], int]:
    # Live quote — direct API call, no retrieval
    if tool_name == "get_stock_quote":
        result_str = _fetch_live_quote(ticker)
        source_counter += 1
        source_metadata[source_counter] = {
            "source_type": "live",
            "document_type": "stock_quote",
            "ticker": ticker,
            "company_name": "",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "section": "Real-time Quote",
            "file_path": "",
            "edgar_url": "",
            "url": "",
        }
        seen_sources[f"{ticker}_live_quote"] = source_counter
        result_str = result_str.replace("[Source: LIVE]", f"[Source {source_counter}]")
        return result_str, [], source_counter

    # Live financial data — selective FMP API calls based on LLM-chosen categories
    if tool_name == "get_financial_data":
        categories = args.get("categories", [])
        result_str, source_counter = _fetch_live_financial_data(ticker, categories, source_counter, seen_sources, source_metadata)
        logger.info(f"Tool get_financial_data({ticker}, categories={categories}) → live FMP data")
        return result_str, [], source_counter

    # Live news — fetch + relevance filter
    if tool_name == "get_news":
        result_str, source_counter = _fetch_live_news(ticker, question, source_counter, seen_sources, source_metadata)
        logger.info(f"Tool get_news({ticker}) → live NewsAPI data (filtered)")
        return result_str, [], source_counter

    # SEC filings — search pre-indexed chunks in ChromaDB
    query = args.get("query", "")
    n_results = max(args.get("n_results", 8), 5)  # Minimum 5 to avoid thin context
    source_types = _TOOL_SOURCE_MAP.get(tool_name)

    chunks = vector_query(
        query_text=query,
        ticker=ticker,
        source_types=source_types,
        n_results=n_results,
    )

    if not chunks:
        logger.warning(
            f"Tool {tool_name}(query='{query[:50]}', n={n_results}) → 0 chunks"
        )
        return (
            "No results found for this query. Try broadening your search "
            "terms or using a different tool.",
            [],
            source_counter,
        )

    formatted, source_counter = _format_tool_results(
        chunks, source_counter, seen_sources, source_metadata
    )

    # Append data boundary based on actual retrieved dates (exclude future/estimate dates)
    today = datetime.now().strftime("%Y-%m-%d")
    dates = sorted(set(
        c["metadata"].get("date", "") for c in chunks
        if c["metadata"].get("date") and c["metadata"]["date"] <= today
    ))
    if dates:
        formatted += (
            f"\n\nDATA BOUNDARY: The above covers ONLY these periods: "
            f"{', '.join(dates)}. Do NOT provide data for any other periods. "
            f"If the user asked for a wider range or data you don't have, "
            f"say what you do have and offer to help within that scope. "
            f"Never mention tools, retrieval, or technical internals."
        )

    logger.info(
        f"Tool {tool_name}(query='{query[:50]}', n={n_results}) → "
        f"{len(chunks)} chunks"
    )

    return formatted, chunks, source_counter


def _fallback_retrieve(
    question: str, ticker: str
) -> tuple[str, list[dict], dict]:
    routing = route_query(question)

    chunks = vector_query(
        query_text=question,
        ticker=ticker,
        source_types=routing["source_types"],
        n_results=routing["n_results"],
    )

    if len(chunks) < 3 and routing["source_types"] is not None:
        chunks = vector_query(
            query_text=question,
            ticker=ticker,
            source_types=None,
            n_results=routing["n_results"],
        )

    source_metadata = {}
    formatted, _ = _format_tool_results(
        chunks, 0, seen_sources={}, source_metadata=source_metadata
    )
    return formatted, chunks, source_metadata


# Strip LLM-generated footnotes, rebuild from actual chunk metadata
def _strip_footnotes(answer: str) -> str:
    lines = answer.rstrip().split("\n")

    cut = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        # Match all common footnote/citation formats the LLM might generate:
        # "[1] ...", "- [1] ...", "• [1] ...", "* [1] ...", "1. [Source] ..."
        if re.match(r"^(\-|\•|\*|\d+\.)?\s*\[(\d+|Source\s*\d*)\]", s):
            cut = i
        elif re.match(r"^#{0,3}\s*(Citations|References|Sources)\s*:?\s*$", s, re.IGNORECASE):
            cut = i
        elif s == "" or s == "---":
            continue
        else:
            break

    body = "\n".join(lines[:cut]).rstrip()
    if body.endswith("---"):
        body = body[:-3].rstrip()

    return body


def _format_footnote(num: int, meta: dict) -> str:
    source_type = meta.get("source_type", "").upper()
    ticker = meta.get("ticker", "")
    doc_type = meta.get("document_type", "").replace("_", " ").title()
    date = meta.get("date", "")
    section = meta.get("section", "")
    edgar_url = meta.get("edgar_url", "")
    url = meta.get("url", "")

    desc = f"{source_type} — {ticker} {doc_type}"
    if date:
        desc += f" ({date})"
    if section:
        desc += f", {section}"

    ref = edgar_url or url
    if ref:
        desc += f" → {ref}"

    return f"- [{num}] {desc}"


def _postprocess_citations(answer: str, source_metadata: dict) -> str:
    if not source_metadata:
        return answer

    body = _strip_footnotes(answer)

    refs_in_order = []
    seen = set()
    for m in re.finditer(r"\[(\d+)\]", body):
        num = int(m.group(1))
        if num not in seen:
            refs_in_order.append(num)
            seen.add(num)

    if not refs_in_order:
        return answer  # No citations to process

    valid_refs = [n for n in refs_in_order if n in source_metadata]

    if not valid_refs:
        return answer

    # Renumber sequentially by first appearance
    remap = {old: new for new, old in enumerate(valid_refs, 1)}

    def _replace_ref(m):
        old = int(m.group(1))
        if old in remap:
            return f"[{remap[old]}]"
        return m.group(0)

    new_body = re.sub(r"\[(\d+)\]", _replace_ref, body)

    # Rebuild footnotes from actual chunk metadata
    footnotes = []
    for old_num in valid_refs:
        new_num = remap[old_num]
        meta = source_metadata[old_num]
        footnotes.append(_format_footnote(new_num, meta))

    citations_block = "\n---\n### Citations\n" + "\n".join(footnotes)
    return new_body.rstrip() + "\n\n" + citations_block



def ask(
    question: str,
    ticker: str,
    show_context: bool = False,
) -> dict:
    ticker = ticker.upper()
    all_chunks = []
    all_context_parts = []
    source_counter = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    tools_called = []
    seen_sources = {}
    source_metadata = {}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Company: {ticker}\n\nQuestion: {question}"},
    ]

    try:
        # Agentic loop — LLM picks tools, max 3 iterations
        for iteration in range(MAX_TOOL_ITERATIONS):
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_completion_tokens=2000,
            )

            msg = response.choices[0].message
            usage = response.usage
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

            if msg.tool_calls:
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    tools_called.append(fn_name)

                    logger.info(f"Tool call [{iteration+1}]: {fn_name}({fn_args})")

                    result_str, chunks, source_counter = _execute_tool(
                        fn_name, fn_args, ticker, question,
                        source_counter, seen_sources, source_metadata,
                    )

                    all_chunks.extend(chunks)
                    all_context_parts.append(result_str)

                    _citation_reminder = (
                        "\n\n[CITATION RULES] Only cite a source for data it actually contains. "
                        "If a value is N/A, null, or missing, say 'not available' with NO citation. "
                        "Never cite sources you searched but found nothing useful in."
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str + _citation_reminder,
                    })
            else:
                answer = msg.content
                break
        else:
            if msg.content:
                answer = msg.content
            else:
                answer = (
                    f"I could only find partial data for {ticker} and wasn't able "
                    f"to fully answer this question. Try narrowing it to a specific metric."
                )

    except Exception as e:
        logger.error(f"Tool calling failed ({e}), falling back to static retrieval")
        # Fallback to static retrieval if tool-calling fails
        context_str, all_chunks, source_metadata = _fallback_retrieve(question, ticker)
        all_context_parts = [context_str]
        tools_called = ["fallback_static"]

        fallback_response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Company: {ticker}\n\nQuestion: {question}\n\n"
                    f"## RETRIEVED CONTEXT\n{context_str}\n\n"
                    f"Answer the question using ONLY the context above."
                )},
            ],
            temperature=0.1,
            max_completion_tokens=2000,
        )
        answer = fallback_response.choices[0].message.content
        fu = fallback_response.usage
        total_prompt_tokens += fu.prompt_tokens
        total_completion_tokens += fu.completion_tokens

    answer = _postprocess_citations(answer, source_metadata)

    full_context = "\n\n---\n\n".join(all_context_parts)
    sources = []
    seen = set()
    for chunk in all_chunks:
        meta = chunk["metadata"]
        key = f"{meta.get('source_type')}_{meta.get('document_type')}_{meta.get('date')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source_type": meta.get("source_type"),
                "document_type": meta.get("document_type"),
                "company": meta.get("company_name", ticker),
                "date": meta.get("date"),
                "section": meta.get("section"),
                "file_path": meta.get("file_path", ""),
                "edgar_url": meta.get("edgar_url", ""),
                "url": meta.get("url", ""),
            })

    # Guardrails: citation check, number verification, LLM grounding
    validation = validate_response(answer, full_context, len(all_chunks))

    if validation["warnings"]:
        logger.warning(f"Guardrail warnings: {validation['warnings']}")

        # One correction pass — feed warnings back to the LLM
        actionable = []
        for w in validation["checks"].get("citation_check", []):
            actionable.append(f"- {w} → Add an inline [N] citation from the sources.")
        for w in validation["checks"].get("number_verification", []):
            actionable.append(f"- {w} → Replace this number with the correct value from the sources, or delete the sentence if no source supports it.")
        for w in validation["checks"].get("grounding_check", []):
            actionable.append(f"- {w} → Delete the specific sentence making this unsupported claim. Keep the rest of the answer intact.")

        if actionable:
            # Strip citations block before feeding to correction — LLM only sees the body
            answer_body = _strip_footnotes(answer)

            correction_prompt = (
                "A fact-checker flagged issues with your answer. Fix ONLY the flagged issues "
                "and return the FULL corrected answer. Keep all correct parts unchanged. "
                "Do NOT add commentary, do NOT describe the corrections, do NOT say what was removed — "
                "just return the corrected answer text.\n\n"
                "Issues:\n" + "\n".join(actionable) + "\n\n"
                "Original answer:\n" + answer_body + "\n\n"
                "Source context:\n" + full_context[:6000]
            )
            try:
                correction = _client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": correction_prompt},
                    ],
                    temperature=0,
                    max_completion_tokens=2000,
                )
                corrected = correction.choices[0].message.content
                cu = correction.usage
                total_prompt_tokens += cu.prompt_tokens
                total_completion_tokens += cu.completion_tokens

                # Sanity check: reject if corrected answer is too short relative to original
                # (prevents the LLM from echoing instructions or gutting the answer)
                if len(corrected) < len(answer_body) * 0.3:
                    logger.info("Guardrail correction too short, keeping original")
                else:
                    corrected = _postprocess_citations(corrected, source_metadata)

                    # Re-validate the corrected answer
                    revalidation = validate_response(corrected, full_context, len(all_chunks))
                    if len(revalidation["warnings"]) < len(validation["warnings"]):
                        answer = corrected
                        validation = revalidation
                        validation["corrected"] = True
                        logger.info("Guardrail correction improved the answer")
                    else:
                        logger.info("Guardrail correction did not improve, keeping original")
            except Exception as e:
                logger.warning(f"Guardrail correction failed ({e}), keeping original")

    total_tokens = total_prompt_tokens + total_completion_tokens

    result = {
        "answer": answer,
        "sources": sources,
        "ticker": ticker,
        "question": question,
        "routing": {
            "method": "tool_calling",
            "tools_called": tools_called,
            "iterations": min(len(set(tools_called)), MAX_TOOL_ITERATIONS),
        },
        "validation": validation,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    if show_context:
        result["context"] = full_context

    return result
