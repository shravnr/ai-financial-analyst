import logging
from datetime import datetime

from src.ingestion.sec_edgar import fetch_sec_filings
from src.ingestion.fmp import fetch_fmp_data
from src.ingestion.news import fetch_news

logger = logging.getLogger(__name__)


def ingest_company(ticker: str) -> dict:
    ticker = ticker.upper()
    logger.info(f"Starting ingestion for {ticker}")

    result = {
        "ticker": ticker,
        "company_name": "",
        "timestamp": datetime.now().isoformat(),
        "sec": None,
        "fmp": None,
        "news": None,
        "summary": {},
    }

    #  1. FMP (run first to get company name for NewsAPI) 
    logger.info(f"[{ticker}] Fetching FMP data...")
    fmp_result = fetch_fmp_data(ticker)
    result["fmp"] = fmp_result
    company_name = fmp_result.get("company_name", "")
    result["company_name"] = company_name

    #  2. SEC EDGAR 
    logger.info(f"[{ticker}] Fetching SEC EDGAR filings...")
    sec_result = fetch_sec_filings(ticker)
    result["sec"] = sec_result

    # Use SEC name if FMP didn't provide one
    if not company_name and sec_result.get("company_name"):
        company_name = sec_result["company_name"]
        result["company_name"] = company_name

    #  3. News 
    if company_name:
        logger.info(f"[{ticker}] Fetching news for '{company_name}'...")
        news_result = fetch_news(company_name, ticker)
    else:
        logger.warning(f"[{ticker}] No company name found, using ticker for news search")
        news_result = fetch_news(ticker, ticker)
    result["news"] = news_result

    #  Build summary 
    sec_filings = sec_result.get("filings", [])
    fmp_files = fmp_result.get("files", {})

    summary = {
        "company_name": company_name,
        "sec_filings": len(sec_filings),
        "sec_10k": sum(1 for f in sec_filings if f["form"] == "10-K"),
        "sec_10q": sum(1 for f in sec_filings if f["form"] == "10-Q"),
        "sec_8k": sum(1 for f in sec_filings if f["form"] == "8-K"),
        "fmp_datasets": len(fmp_files),
        "news_articles": news_result.get("article_count", 0),
        "errors": (
            sec_result.get("errors", [])
            + fmp_result.get("errors", [])
            + news_result.get("errors", [])
        ),
    }
    result["summary"] = summary

    logger.info(
        f"[{ticker}] Ingestion complete: "
        f"{summary['sec_filings']} SEC filings "
        f"({summary['sec_10k']} 10-K, {summary['sec_10q']} 10-Q, {summary['sec_8k']} 8-K), "
        f"{summary['fmp_datasets']} FMP datasets, "
        f"{summary['news_articles']} news articles"
    )
    if summary["errors"]:
        logger.warning(f"[{ticker}] Errors: {summary['errors']}")

    return result
