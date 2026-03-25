import json
import logging
import time
from pathlib import Path

import requests

from src.config import FMP_API_KEY, FMP_BASE_URL, RAW_DATA_DIR

logger = logging.getLogger(__name__)


_FMP_MAX_RETRIES = 2
_FMP_RETRY_DELAY = 1.5  # seconds

# Sentinels to distinguish error types from "no data"
FMP_RATE_LIMITED = "FMP_RATE_LIMITED"
FMP_PAYMENT_REQUIRED = "FMP_PAYMENT_REQUIRED"


def _fmp_get(endpoint: str, params: dict | None = None) -> list | dict | str | None:
    # Returns data on success, None on empty/error, FMP_RATE_LIMITED on 429, FMP_PAYMENT_REQUIRED on 402
    if not FMP_API_KEY:
        logger.error("FMP_API_KEY not set")
        return None

    params = params or {}
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE_URL}/{endpoint}"

    for attempt in range(_FMP_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=20)

            if resp.status_code == 429:
                if attempt < _FMP_MAX_RETRIES:
                    delay = _FMP_RETRY_DELAY * (attempt + 1)
                    logger.info(f"FMP rate limited on {endpoint}, retrying in {delay}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(f"FMP rate limited on {endpoint}, retries exhausted")
                    return FMP_RATE_LIMITED

            if resp.status_code == 402:
                logger.info(f"FMP endpoint {endpoint} not available on current plan for {params.get('symbol', '?')}")
                return FMP_PAYMENT_REQUIRED

            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and ("Error Message" in data or "error" in data):
                msg = data.get("Error Message") or data.get("error", "Unknown error")
                logger.warning(f"FMP error for {endpoint}: {msg}")
                return None

            return data
        except requests.RequestException as e:
            if attempt < _FMP_MAX_RETRIES and "429" in str(e):
                delay = _FMP_RETRY_DELAY * (attempt + 1)
                logger.info(f"FMP rate limited on {endpoint}, retrying in {delay}s")
                time.sleep(delay)
                continue
            logger.warning(f"FMP request failed for {endpoint}: {e}")
            return None

    return None


def _save_json(data, ticker: str, filename: str) -> str | None:
    if not data:
        return None

    save_dir = RAW_DATA_DIR / ticker / "fmp"
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return str(filepath)



def _fetch_profile(ticker: str) -> dict | None:
    data = _fmp_get("profile", {"symbol": ticker})
    if data and isinstance(data, list) and len(data) > 0:
        return data[0]
    return None


def fetch_fmp_profile(ticker: str) -> dict:
    """Fetch and save only the company profile (for company name during SEC ingestion)."""
    ticker = ticker.upper()
    result = {"company_name": "", "errors": []}

    profile = _fetch_profile(ticker)
    if not profile:
        result["errors"].append(f"Company profile not found for {ticker} on FMP")
        return result

    result["company_name"] = profile.get("companyName", "")
    _save_json([profile], ticker, "profile.json")
    return result


