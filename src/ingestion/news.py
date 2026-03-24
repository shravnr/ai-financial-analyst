import json
import logging
from pathlib import Path

import requests

from src.config import NEWS_API_KEY, NEWS_API_BASE_URL, RAW_DATA_DIR

logger = logging.getLogger(__name__)


def fetch_news(
    company_name: str,
    ticker: str,
) -> dict:
    ticker = ticker.upper()
    result = {"article_count": 0, "file_path": None, "errors": []}

    if not NEWS_API_KEY:
        result["errors"].append("NEWS_API_KEY not set")
        return result

    # Build query: strip common corporate suffixes for cleaner search
    query = company_name
    for suffix in [" Inc.", " Inc", " Corp.", " Corp", " Ltd.", " Ltd",
                   " plc", " PLC", " Co.", " Co", " Corporation",
                   " Incorporated", " Limited"]:
        query = query.replace(suffix, "")
    query = query.strip()

    if not query:
        query = ticker  # Fallback to ticker if name is empty

    try:
        resp = requests.get(
            f"{NEWS_API_BASE_URL}/everything",
            params={
                "q": f'"{query}"',  # Exact phrase match
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": NEWS_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.RequestException as e:
        result["errors"].append(f"NewsAPI request failed: {e}")
        return result

    if data.get("status") != "ok":
        result["errors"].append(f"NewsAPI error: {data.get('message', 'Unknown')}")
        return result

    articles = data.get("articles", [])
    result["article_count"] = len(articles)

    if not articles:
        result["errors"].append(f"No news articles found for '{query}'")
        return result

    # Save raw response
    save_dir = RAW_DATA_DIR / ticker / "news"
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / "articles.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, default=str)

    result["file_path"] = str(filepath)
    logger.info(f"Saved {len(articles)} news articles for {ticker}")

    return result
