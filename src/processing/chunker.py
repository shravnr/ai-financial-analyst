import re
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── 10-K section detection ────────────────────────────────────────────

# Standard 10-K Item patterns — ordered by appearance in filings
_10K_SECTIONS = [
    (r"(?i)\bItem\s+1A[\.\s\-\—]+Risk\s+Factors", "Item 1A - Risk Factors"),
    (r"(?i)\bItem\s+1B", "Item 1B - Unresolved Staff Comments"),
    (r"(?i)\bItem\s+1C", "Item 1C - Cybersecurity"),
    (r"(?i)\bItem\s+1[\.\s\-\—]+Business", "Item 1 - Business"),
    (r"(?i)\bItem\s+2[\.\s\-\—]+Propert", "Item 2 - Properties"),
    (r"(?i)\bItem\s+3[\.\s\-\—]+Legal", "Item 3 - Legal Proceedings"),
    (r"(?i)\bItem\s+5", "Item 5 - Market for Common Equity"),
    (r"(?i)\bItem\s+7A", "Item 7A - Market Risk Disclosures"),
    (r"(?i)\bItem\s+7[\.\s\-\—]+Management", "Item 7 - MD&A"),
    (r"(?i)\bItem\s+8[\.\s\-\—]+Financial\s+Statements", "Item 8 - Financial Statements"),
    (r"(?i)\bItem\s+9A", "Item 9A - Controls and Procedures"),
    (r"(?i)\bItem\s+9[\.\s\-\—]", "Item 9"),
    (r"(?i)\bItem\s+10", "Item 10 - Directors and Officers"),
    (r"(?i)\bItem\s+11", "Item 11 - Executive Compensation"),
    (r"(?i)\bItem\s+12", "Item 12 - Security Ownership"),
    (r"(?i)\bItem\s+13", "Item 13 - Certain Relationships"),
    (r"(?i)\bItem\s+14", "Item 14 - Principal Accountant Fees"),
    (r"(?i)\bItem\s+15", "Item 15 - Exhibits"),
]


def _detect_section(text: str, position: int, full_text: str) -> str:
    # Look at text before this chunk's position
    preceding = full_text[:position]

    best_section = "Preamble"
    best_pos = -1

    for pattern, section_name in _10K_SECTIONS:
        for match in re.finditer(pattern, preceding):
            if match.start() > best_pos:
                best_pos = match.start()
                best_section = section_name

    return best_section


# ── Chunking ──────────────────────────────────────────────────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=500,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def chunk_sec_filing(
    text: str,
    metadata: dict,
) -> list[dict]:
    is_10k = metadata.get("document_type") == "10-K"

    # For small documents (8-K), don't split
    if len(text) < 4000:
        section = "Earnings Press Release" if metadata.get("document_type") == "8-K" else "Full Document"
        return [{
            "text": text,
            "metadata": {**metadata, "chunk_index": 0, "section": section},
        }]

    chunks = _splitter.split_text(text)

    # Track character positions for section detection
    results = []
    search_start = 0
    for i, chunk_text in enumerate(chunks):
        # Find where this chunk starts in the original text
        pos = text.find(chunk_text[:100], search_start)
        if pos == -1:
            pos = search_start

        chunk_metadata = {**metadata, "chunk_index": i}

        if is_10k:
            chunk_metadata["section"] = _detect_section(chunk_text, pos, text)
        else:
            chunk_metadata["section"] = f"Part {i + 1}"

        results.append({"text": chunk_text, "metadata": chunk_metadata})
        search_start = max(search_start, pos + 1)

    logger.info(
        f"Chunked {metadata.get('document_type', '?')} "
        f"({metadata.get('date', '?')}): {len(text):,} chars → {len(chunks)} chunks"
    )
    return results


def chunk_news_articles(articles: list[dict], ticker: str, company_name: str) -> list[dict]:
    results = []
    for i, article in enumerate(articles):
        # Build text from available fields
        parts = []
        title = article.get("title", "")
        if title:
            parts.append(title)
        desc = article.get("description", "")
        if desc:
            parts.append(desc)
        content = article.get("content", "")
        if content:
            # NewsAPI free tier truncates content with "[+N chars]" suffix
            content = re.sub(r"\[\+\d+ chars\]$", "", content).strip()
            if content:
                parts.append(content)

        text = "\n\n".join(parts)
        if not text.strip():
            continue

        source_name = article.get("source", {}).get("name", "Unknown")
        pub_date = article.get("publishedAt", "")[:10]  # YYYY-MM-DD

        results.append({
            "text": text,
            "metadata": {
                "ticker": ticker,
                "company_name": company_name,
                "source_type": "news",
                "document_type": "news_article",
                "date": pub_date,
                "section": f"News - {source_name}",
                "source_name": source_name,
                "url": article.get("url", ""),
                "chunk_index": i,
            },
        })

    logger.info(f"Processed {len(results)} news articles for {ticker}")
    return results
