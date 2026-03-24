import json
import logging

from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL_MINI

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

#  Fast path: common company names → tickers 

_COMMON_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
    "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA",
    "netflix": "NFLX", "berkshire": "BRK-B", "jpmorgan": "JPM",
    "jp morgan": "JPM", "johnson & johnson": "JNJ", "j&j": "JNJ",
    "walmart": "WMT", "visa": "V", "mastercard": "MA",
    "procter & gamble": "PG", "p&g": "PG", "unitedhealth": "UNH",
    "home depot": "HD", "disney": "DIS", "walt disney": "DIS",
    "coca-cola": "KO", "coca cola": "KO", "pepsi": "PEP",
    "pepsico": "PEP", "intel": "INTC", "amd": "AMD",
    "salesforce": "CRM", "adobe": "ADBE", "paypal": "PYPL",
    "boeing": "BA", "chevron": "CVX", "exxon": "XOM",
    "exxonmobil": "XOM", "goldman sachs": "GS", "morgan stanley": "MS",
    "citigroup": "C", "bank of america": "BAC",
}

_LLM_RESOLVER_SYSTEM = """You are a ticker symbol resolver. Given a user's question about a company, extract the company name and return its US stock ticker symbol.

Rules:
- If the question mentions a specific company, return its ticker
- If the question already contains a ticker symbol (e.g., "AAPL", "MSFT"), return it directly
- If no company is mentioned, return null
- Only return US-listed company tickers

Return JSON: {"ticker": "AAPL", "company": "Apple Inc."} or {"ticker": null, "company": null}"""


def resolve_ticker(question: str) -> str | None:
    
    q_upper = question.upper()
    q_lower = question.lower()

    #  Tier 1: Direct ticker mention 
    # Check if any known ticker appears as a standalone word
    import re
    common_tickers = set(_COMMON_TICKERS.values())
    for ticker in common_tickers:
        if re.search(rf"\b{re.escape(ticker)}\b", q_upper):
            logger.debug(f"Direct ticker match: {ticker}")
            return ticker

    #  Tier 2: Common company names ─
    for name, ticker in _COMMON_TICKERS.items():
        if name in q_lower:
            logger.debug(f"Common name match: '{name}' → {ticker}")
            return ticker

    #  Tier 3: LLM resolution ─
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL_MINI,
            messages=[
                {"role": "system", "content": _LLM_RESOLVER_SYSTEM},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=50,
        )

        result = json.loads(response.choices[0].message.content)
        ticker = result.get("ticker")

        if ticker:
            ticker = ticker.upper().strip()
            company = result.get("company", "")
            logger.info(f"LLM resolved: '{company}' → {ticker}")
            return ticker

        logger.debug("LLM found no company in question")
        return None

    except Exception as e:
        logger.warning(f"Ticker resolution failed ({e})")
        return None
