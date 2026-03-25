import re
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# SEC filing section patterns — ordered by appearance in filings
# These patterns work for both 10-K and 10-Q (which share Item numbering)
_SEC_SECTIONS = [
    # Common to 10-K and 10-Q
    (r"(?i)\bItem\s+1A[\.\s\-\—]+Risk\s+Factors", "Item 1A - Risk Factors"),
    (r"(?i)\bItem\s+1B", "Item 1B - Unresolved Staff Comments"),
    (r"(?i)\bItem\s+1C", "Item 1C - Cybersecurity"),
    (r"(?i)\bItem\s+1[\.\s\-\—]+Business", "Item 1 - Business"),
    (r"(?i)\bItem\s+1[\.\s\-\—]+Financial\s+Statements", "Item 1 - Financial Statements"),
    (r"(?i)\bItem\s+2[\.\s\-\—]+Propert", "Item 2 - Properties"),
    (r"(?i)\bItem\s+2[\.\s\-\—]+Management", "Item 2 - MD&A"),
    (r"(?i)\bItem\s+3[\.\s\-\—]+Legal", "Item 3 - Legal Proceedings"),
    (r"(?i)\bItem\s+3[\.\s\-\—]+Quantitative", "Item 3 - Market Risk Disclosures"),
    (r"(?i)\bItem\s+4[\.\s\-\—]+Controls", "Item 4 - Controls and Procedures"),
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
    # 10-Q Part headers
    (r"(?i)\bPart\s+I[\.\s\-\—]+Financial\s+Information", "Part I - Financial Information"),
    (r"(?i)\bPart\s+II[\.\s\-\—]+Other\s+Information", "Part II - Other Information"),
    # Fallback content patterns
    (r"(?i)\bManagement.s\s+Discussion\s+and\s+Analysis", "MD&A"),
    (r"(?i)\bRisk\s+Factors", "Risk Factors"),
    (r"(?i)\bFinancial\s+Statements", "Financial Statements"),
    (r"(?i)\bNotes\s+to\s+(Consolidated\s+)?Financial\s+Statements", "Notes to Financial Statements"),
]


def _detect_section(text: str, position: int, full_text: str) -> str:
    preceding = full_text[:position]

    best_section = "Preamble"
    best_pos = -1

    for pattern, section_name in _SEC_SECTIONS:
        for match in re.finditer(pattern, preceding):
            if match.start() > best_pos:
                best_pos = match.start()
                best_section = section_name

    return best_section


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
    # Small docs (8-K <4KB) stay whole
    if len(text) < 4000:
        section = "Earnings Press Release" if metadata.get("document_type") == "8-K" else "Full Document"
        return [{
            "text": text,
            "metadata": {**metadata, "chunk_index": 0, "section": section},
        }]

    chunks = _splitter.split_text(text)

    results = []
    search_start = 0
    for i, chunk_text in enumerate(chunks):
        pos = text.find(chunk_text[:100], search_start)
        if pos == -1:
            pos = search_start

        chunk_metadata = {**metadata, "chunk_index": i}
        chunk_metadata["section"] = _detect_section(chunk_text, pos, text)

        results.append({"text": chunk_text, "metadata": chunk_metadata})
        search_start = max(search_start, pos + 1)

    logger.info(
        f"Chunked {metadata.get('document_type', '?')} "
        f"({metadata.get('date', '?')}): {len(text):,} chars → {len(chunks)} chunks"
    )
    return results


