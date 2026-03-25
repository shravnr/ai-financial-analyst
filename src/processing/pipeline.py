import json
import logging
from pathlib import Path

from src.config import RAW_DATA_DIR, PROJECT_ROOT
from src.processing.chunker import chunk_sec_filing
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


def process_company(ticker: str, reprocess: bool = False) -> dict:
    ticker = ticker.upper()
    raw_dir = RAW_DATA_DIR / ticker

    if not raw_dir.exists():
        return {
            "ticker": ticker,
            "error": f"No raw data found for {ticker}. Run ingestion first.",
            "chunks": {"sec": 0, "total": 0},
        }

    company_name = _load_company_name(ticker, raw_dir)
    logger.info(f"Processing {ticker} ({company_name})")

    if reprocess:
        deleted = delete_company(ticker)
        logger.info(f"Deleted {deleted} existing documents for {ticker}")

    sec_chunks = _process_sec_filings(ticker, company_name, raw_dir)

    if sec_chunks:
        added = add_documents(sec_chunks)
    else:
        added = 0

    stats = {
        "ticker": ticker,
        "company_name": company_name,
        "chunks": {
            "sec": len(sec_chunks),
            "total": len(sec_chunks),
        },
        "embedded": added,
    }

    logger.info(
        f"Processed {ticker}: {stats['chunks']['sec']} SEC chunks — "
        f"{added} total embedded"
    )

    return stats
