import json
import logging
from pathlib import Path

import requests

from src.config import FMP_API_KEY, FMP_BASE_URL, RAW_DATA_DIR

logger = logging.getLogger(__name__)


#  Helpers 

def _fmp_get(endpoint: str, params: dict | None = None) -> list | dict | None:
    if not FMP_API_KEY:
        logger.error("FMP_API_KEY not set")
        return None

    params = params or {}
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE_URL}/{endpoint}"

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # FMP returns error messages as dicts
        if isinstance(data, dict) and ("Error Message" in data or "error" in data):
            msg = data.get("Error Message") or data.get("error", "Unknown error")
            logger.warning(f"FMP error for {endpoint}: {msg}")
            return None

        return data
    except requests.RequestException as e:
        logger.warning(f"FMP request failed for {endpoint}: {e}")
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


#  Individual fetchers 

def _fetch_profile(ticker: str) -> dict | None:
    data = _fmp_get("profile", {"symbol": ticker})
    if data and isinstance(data, list) and len(data) > 0:
        return data[0]
    return None


def _fetch_financial_statements(ticker: str) -> dict:
    results = {}

    endpoints = {
        "income_statement_annual": ("income-statement", {"symbol": ticker, "period": "annual", "limit": 2}),
        "income_statement_quarterly": ("income-statement", {"symbol": ticker, "period": "quarter", "limit": 8}),
        "balance_sheet_annual": ("balance-sheet-statement", {"symbol": ticker, "period": "annual", "limit": 2}),
        "balance_sheet_quarterly": ("balance-sheet-statement", {"symbol": ticker, "period": "quarter", "limit": 8}),
        "cash_flow_annual": ("cash-flow-statement", {"symbol": ticker, "period": "annual", "limit": 2}),
        "cash_flow_quarterly": ("cash-flow-statement", {"symbol": ticker, "period": "quarter", "limit": 8}),
    }

    for name, (endpoint, params) in endpoints.items():
        data = _fmp_get(endpoint, params)
        results[name] = data
        if data:
            logger.info(f"Fetched {name}: {len(data)} records")
        else:
            logger.warning(f"No data for {name}")

    return results


def _fetch_metrics_and_ratios(ticker: str) -> dict:
    results = {}

    endpoints = {
        "key_metrics": ("key-metrics", {"symbol": ticker, "period": "annual", "limit": 2}),
        "ratios": ("ratios", {"symbol": ticker, "period": "annual", "limit": 2}),
        "grades": ("grades", {"symbol": ticker, "limit": 20}),
        "analyst_estimates": ("analyst-estimates", {"symbol": ticker, "period": "annual", "limit": 4}),
    }

    for name, (endpoint, params) in endpoints.items():
        data = _fmp_get(endpoint, params)
        results[name] = data
        if data:
            count = len(data) if isinstance(data, list) else 1
            logger.info(f"Fetched {name}: {count} records")
        else:
            logger.warning(f"No data for {name}")

    return results


#  Public API 

def fetch_fmp_data(ticker: str) -> dict:
    ticker = ticker.upper()
    result = {"company_name": "", "profile": None, "files": {}, "errors": []}

    # 1. Company profile
    profile = _fetch_profile(ticker)
    if not profile:
        result["errors"].append(f"Company profile not found for {ticker} on FMP")
        return result

    result["company_name"] = profile.get("companyName", "")
    result["profile"] = profile
    path = _save_json([profile], ticker, "profile.json")
    if path:
        result["files"]["profile"] = path

    # 2. Financial statements
    statements = _fetch_financial_statements(ticker)
    for name, data in statements.items():
        path = _save_json(data, ticker, f"{name}.json")
        if path:
            result["files"][name] = path
        elif data is None:
            result["errors"].append(f"Failed to fetch {name}")

    # 3. Key metrics, ratios, analyst data
    metrics = _fetch_metrics_and_ratios(ticker)
    for name, data in metrics.items():
        path = _save_json(data, ticker, f"{name}.json")
        if path:
            result["files"][name] = path
        elif data is None:
            result["errors"].append(f"Failed to fetch {name}")

    return result
