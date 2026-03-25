import logging
from datetime import datetime

from src.ingestion.sec_edgar import fetch_sec_filings
from src.ingestion.fmp import fetch_fmp_profile

logger = logging.getLogger(__name__)


def ingest_company(ticker: str) -> dict:
    ticker = ticker.upper()
    logger.info(f"Starting ingestion for {ticker}")

    result = {
        "ticker": ticker,
        "company_name": "",
        "timestamp": datetime.now().isoformat(),
        "sec": None,
        "summary": {},
    }

    # Fetch FMP profile only — just need company name
    logger.info(f"[{ticker}] Fetching company profile...")
    profile_result = fetch_fmp_profile(ticker)
    company_name = profile_result.get("company_name", "")
    result["company_name"] = company_name

    # SEC filings — these get chunked and indexed
    logger.info(f"[{ticker}] Fetching SEC EDGAR filings...")
    sec_result = fetch_sec_filings(ticker)
    result["sec"] = sec_result

    if not company_name and sec_result.get("company_name"):
        company_name = sec_result["company_name"]
        result["company_name"] = company_name

    sec_filings = sec_result.get("filings", [])

    summary = {
        "company_name": company_name,
        "sec_filings": len(sec_filings),
        "sec_10k": sum(1 for f in sec_filings if f["form"] == "10-K"),
        "sec_10q": sum(1 for f in sec_filings if f["form"] == "10-Q"),
        "sec_8k": sum(1 for f in sec_filings if f["form"] == "8-K"),
        "errors": sec_result.get("errors", []) + profile_result.get("errors", []),
    }
    result["summary"] = summary

    logger.info(
        f"[{ticker}] Ingestion complete: "
        f"{summary['sec_filings']} SEC filings "
        f"({summary['sec_10k']} 10-K, {summary['sec_10q']} 10-Q, {summary['sec_8k']} 8-K)"
    )
    if summary["errors"]:
        logger.warning(f"[{ticker}] Errors: {summary['errors']}")

    return result
