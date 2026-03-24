import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from src.config import (
    RAW_DATA_DIR,
    SEC_BASE_URL,
    SEC_EDGAR_USER_AGENT,
    SEC_RATE_LIMIT_DELAY,
    FILING_YEARS_BACK,
)

logger = logging.getLogger(__name__)

#  Helpers 

_session = requests.Session()
_session.headers.update({
    "User-Agent": SEC_EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
})

_last_request_time = 0.0


def _rate_limited_get(url: str, **kwargs) -> requests.Response:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < SEC_RATE_LIMIT_DELAY:
        time.sleep(SEC_RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()

    resp = _session.get(url, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp


#  Ticker → CIK 

_cik_cache: dict[str, str] = {}


def _load_cik_mapping() -> dict[str, str]:
    if _cik_cache:
        return _cik_cache

    url = "https://www.sec.gov/files/company_tickers.json"
    resp = _rate_limited_get(url)
    data = resp.json()

    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        _cik_cache[ticker] = cik

    logger.info(f"Loaded {len(_cik_cache)} ticker-CIK mappings from EDGAR")
    return _cik_cache


def ticker_to_cik(ticker: str) -> str | None:
    """Return zero-padded CIK for a ticker, or None if not found."""
    mapping = _load_cik_mapping()
    return mapping.get(ticker.upper())


#  Filings metadata 

def _get_filings_metadata(cik: str) -> dict:
    url = f"{SEC_BASE_URL}/submissions/CIK{cik}.json"
    resp = _rate_limited_get(url)
    return resp.json()


def _filter_filings(
    metadata: dict,
    form_types: list[str],
    years_back: int,
) -> list[dict]:
    cutoff = datetime.now() - timedelta(days=years_back * 365)
    recent = metadata.get("filings", {}).get("recent", {})

    if not recent:
        return []

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    for i, form in enumerate(forms):
        if form not in form_types:
            continue
        filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
        if filing_date < cutoff:
            continue
        results.append({
            "form": form,
            "filingDate": dates[i],
            "accessionNumber": accessions[i],
            "primaryDocument": primary_docs[i],
        })

    return results


#  Filing text extraction 

def _fetch_filing_text(cik: str, filing: dict) -> str:
    accession_no_dashes = filing["accessionNumber"].replace("-", "")
    primary_doc = filing["primaryDocument"]
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik.lstrip('0')}/{accession_no_dashes}/{primary_doc}"
    )

    logger.info(f"Fetching filing: {filing['form']} {filing['filingDate']} from {url}")
    resp = _rate_limited_get(url)

    content_type = resp.headers.get("Content-Type", "")
    if "html" in content_type or primary_doc.endswith((".htm", ".html")):
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = resp.text

    # Normalize whitespace: collapse blank lines, strip trailing spaces
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False

    return "\n".join(cleaned)


#  Public API 

def fetch_sec_filings(
    ticker: str,
    form_types: list[str] | None = None,
    years_back: int | None = None,
) -> dict:
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]
    if years_back is None:
        years_back = FILING_YEARS_BACK

    ticker = ticker.upper()
    result = {"company_name": "", "cik": "", "filings": [], "errors": []}

    # Resolve CIK
    cik = ticker_to_cik(ticker)
    if not cik:
        result["errors"].append(f"Ticker '{ticker}' not found in EDGAR")
        return result
    result["cik"] = cik

    # Get filings metadata
    try:
        metadata = _get_filings_metadata(cik)
    except requests.RequestException as e:
        result["errors"].append(f"Failed to fetch EDGAR metadata: {e}")
        return result

    result["company_name"] = metadata.get("name", "")

    # Filter to desired form types and date range
    filings = _filter_filings(metadata, form_types, years_back)
    if not filings:
        result["errors"].append(
            f"No {form_types} filings found for {ticker} in the last {years_back} years"
        )
        return result

    logger.info(f"Found {len(filings)} filings for {ticker}: "
                f"{[f'{f['form']} ({f['filingDate']})' for f in filings]}")

    # Fetch and store each filing
    save_dir = RAW_DATA_DIR / ticker / "sec"
    save_dir.mkdir(parents=True, exist_ok=True)

    for filing in filings:
        try:
            text = _fetch_filing_text(cik, filing)
            # Save with descriptive filename
            filename = f"{filing['filingDate']}_{filing['form']}.txt"
            filepath = save_dir / filename
            filepath.write_text(text, encoding="utf-8")

            result["filings"].append({
                "form": filing["form"],
                "date": filing["filingDate"],
                "path": str(filepath),
                "char_count": len(text),
                "accession_number": filing["accessionNumber"],
                "primary_document": filing["primaryDocument"],
            })
            logger.info(
                f"Saved {filing['form']} ({filing['filingDate']}): "
                f"{len(text):,} chars → {filepath.name}"
            )
        except Exception as e:
            msg = f"Failed to fetch {filing['form']} ({filing['filingDate']}): {e}"
            logger.warning(msg)
            result["errors"].append(msg)

    # Save filing metadata for processing pipeline (EDGAR URLs for citations)
    if result["filings"]:
        filings_meta = {}
        for f in result["filings"]:
            filename = f"{f['date']}_{f['form']}.txt"
            acc_no_dashes = f["accession_number"].replace("-", "")
            edgar_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik.lstrip('0')}/{acc_no_dashes}/{f['primary_document']}"
            )
            filings_meta[filename] = {"edgar_url": edgar_url}
        meta_path = save_dir / "filings_meta.json"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(filings_meta, mf, indent=2)

    return result
