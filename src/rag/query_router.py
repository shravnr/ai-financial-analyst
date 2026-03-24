import json
import re
import logging

from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL_MINI

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

#  Regex fast path 
# Keyword patterns mapped to source preferences
_PATTERNS: list[tuple[str, list[str], int]] = [
    # (regex pattern, preferred source_types, n_results)

    # Pure structured data questions → prioritize FMP
    (r"(?i)(revenue|profit|income|earnings|eps|margin|ebitda)\b.*\b(number|figure|exact|how much|what)",
     ["fmp", "sec"], 12),

    # News/recent events → prioritize news + 8-K
    (r"(?i)(latest|recent|news|today|this week|this month|announce|event)",
     ["news", "sec"], 12),

    # Risk questions → prioritize SEC 10-K
    (r"(?i)(risk|threat|challenge|headwind|concern|uncertainty|litigation|lawsuit)",
     ["sec"], 12),

    # Management guidance/strategy → SEC filings
    (r"(?i)(guidance|outlook|forecast|forward|strategy|plan|initiative|management said|going forward)",
     ["sec", "fmp"], 12),

    # Debt/cash/balance sheet → FMP + SEC
    (r"(?i)(debt|leverage|liabilit|borrow|loan|credit|bond|cash position|cash flow|liquidity|balance sheet)",
     ["fmp", "sec"], 12),

    # Valuation → FMP metrics
    (r"(?i)(valuation|overvalued|undervalued|p\/e|pe ratio|price.to|ev\/|enterprise value|fair value)",
     ["fmp"], 10),

    # Analyst ratings → FMP grades
    (r"(?i)(analyst|rating|upgrade|downgrade|buy|sell|hold|target price|consensus|estimate)",
     ["fmp"], 10),

    # Competitor comparison → all sources
    (r"(?i)(competitor|compar|versus|vs\.?|peer|rival|industry|sector)",
     None, 12),  # None = all sources

    # Segment/business breakdown → SEC filings
    (r"(?i)(segment|division|business line|revenue driver|breakdown|geographic|product mix)",
     ["sec", "fmp"], 12),

    # Earnings call / earnings summary → SEC 8-K + FMP
    (r"(?i)(earnings call|earnings report|quarterly result|quarter result|conference call)",
     ["sec", "fmp"], 12),
]

#  LLM classification prompt 

_LLM_ROUTER_SYSTEM = """You are a financial question classifier. Given a user's question about a company, determine which data sources to search.

Available sources:
- "sec": SEC filings (10-K annual reports, 10-Q quarterly reports, 8-K earnings press releases). Best for: risks, strategy, business segments, legal matters, management discussion, qualitative analysis.
- "fmp": Financial Modeling Prep structured data (income statements, balance sheets, cash flow, key metrics, ratios, analyst grades, estimates). Best for: specific numbers, ratios, valuation, analyst ratings, financial trends.
- "news": Recent news articles. Best for: latest developments, current events, market sentiment.

Return JSON with:
- "source_types": list of 1-3 source strings, or null to search all sources
- "n_results": number of chunks to retrieve (8-15, use 12 for broad questions, 8-10 for focused ones)
- "reasoning": one sentence explaining your choice

Example: {"source_types": ["fmp", "sec"], "n_results": 12, "reasoning": "Revenue comparison needs financial statements from FMP and SEC filings for context."}"""


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
            max_tokens=150,
        )

        result = json.loads(response.choices[0].message.content)
        source_types = result.get("source_types")
        n_results = result.get("n_results", 12)
        reasoning = result.get("reasoning", "")

        # Validate source_types
        valid_sources = {"sec", "fmp", "news"}
        if source_types is not None:
            source_types = [s for s in source_types if s in valid_sources]
            if not source_types:
                source_types = None  # fall back to all

        # Clamp n_results
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
    #  Tier 1: Regex fast path 
    for pattern, sources, n_results in _PATTERNS:
        if re.search(pattern, question):
            logger.debug(f"Regex matched: {pattern[:50]}... → sources={sources}")
            return {
                "source_types": sources,
                "n_results": n_results,
                "matched_pattern": pattern[:50],
                "router_type": "regex",
            }

    #  Tier 2: LLM classification 
    logger.info(f"No regex match for: '{question[:80]}' — escalating to LLM router")
    return _llm_classify(question)
