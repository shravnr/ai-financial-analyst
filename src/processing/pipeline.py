import json
import logging
from pathlib import Path

from src.config import RAW_DATA_DIR, PROJECT_ROOT
from src.processing.chunker import chunk_sec_filing, chunk_news_articles
from src.processing.structured_formatter import format_fmp_data
from src.processing.vector_store import add_documents, delete_company

logger = logging.getLogger(__name__)


def _load_company_name(ticker: str, raw_dir: Path) -> str:
    profile_path = raw_dir / "fmp" / "profile.json"
    if profile_path.exists():
        try:
            with open(profile_path) as f:
                data = json.load(f)
                record = data[0] if isinstance(data, list) else data
                return record.get("companyName", ticker)
        except Exception:
            pass
    return ticker


def _process_sec_filings(ticker: str, company_name: str, raw_dir: Path) -> list[dict]:
    sec_dir = raw_dir / "sec"
    if not sec_dir.exists():
        logger.warning(f"No SEC data directory for {ticker}")
        return []

    filings_meta = {}
    meta_path = sec_dir / "filings_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            filings_meta = json.load(f)

    all_chunks = []
    for filepath in sorted(sec_dir.glob("*.txt")):
        stem = filepath.stem  # e.g., "2025-10-31_10-K"
        parts = stem.split("_", 1)
        if len(parts) == 2:
            date, form_type = parts
        else:
            date, form_type = "unknown", stem

        text = filepath.read_text(encoding="utf-8")

        file_meta = filings_meta.get(filepath.name, {})

        metadata = {
            "ticker": ticker,
            "company_name": company_name,
            "source_type": "sec",
            "document_type": form_type,
            "date": date,
            "file_path": str(filepath.relative_to(PROJECT_ROOT)),
            "edgar_url": file_meta.get("edgar_url", ""),
        }

        chunks = chunk_sec_filing(text, metadata)
        all_chunks.extend(chunks)

    logger.info(f"SEC filings for {ticker}: {len(all_chunks)} chunks")
    return all_chunks


def _process_news(ticker: str, company_name: str, raw_dir: Path) -> list[dict]:
    news_file = raw_dir / "news" / "articles.json"
    if not news_file.exists():
        logger.warning(f"No news data for {ticker}")
        return []

    with open(news_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    chunks = chunk_news_articles(articles, ticker, company_name)
    logger.info(f"News for {ticker}: {len(chunks)} article chunks")
    return chunks


def process_company(ticker: str, reprocess: bool = False) -> dict:
    ticker = ticker.upper()
    raw_dir = RAW_DATA_DIR / ticker

    if not raw_dir.exists():
        return {
            "ticker": ticker,
            "error": f"No raw data found for {ticker}. Run ingestion first.",
            "chunks": {"sec": 0, "fmp": 0, "news": 0, "total": 0},
        }

    company_name = _load_company_name(ticker, raw_dir)
    logger.info(f"Processing {ticker} ({company_name})")

    if reprocess:
        deleted = delete_company(ticker)
        logger.info(f"Deleted {deleted} existing documents for {ticker}")

    sec_chunks = _process_sec_filings(ticker, company_name, raw_dir)
    fmp_chunks = format_fmp_data(ticker, raw_dir)
    news_chunks = _process_news(ticker, company_name, raw_dir)

    all_chunks = sec_chunks + fmp_chunks + news_chunks

    if all_chunks:
        added = add_documents(all_chunks)
    else:
        added = 0

    stats = {
        "ticker": ticker,
        "company_name": company_name,
        "chunks": {
            "sec": len(sec_chunks),
            "fmp": len(fmp_chunks),
            "news": len(news_chunks),
            "total": len(all_chunks),
        },
        "embedded": added,
    }

    logger.info(
        f"Processed {ticker}: {stats['chunks']['sec']} SEC chunks, "
        f"{stats['chunks']['fmp']} FMP docs, "
        f"{stats['chunks']['news']} news chunks — "
        f"{added} total embedded"
    )

    return stats
