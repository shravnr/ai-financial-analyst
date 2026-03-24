import logging

from src.processing.vector_store import query as vector_query
from src.rag.query_router import route_query

logger = logging.getLogger(__name__)

_MIN_RESULTS = 3


def retrieve(question: str, ticker: str) -> dict:
    ticker = ticker.upper()
    routing = route_query(question)

    # Primary retrieval with routed sources
    chunks = vector_query(
        query_text=question,
        ticker=ticker,
        source_types=routing["source_types"],
        n_results=routing["n_results"],
    )

    # Fallback: if too few results, broaden to all sources
    if len(chunks) < _MIN_RESULTS and routing["source_types"] is not None:
        logger.info(
            f"Only {len(chunks)} results from {routing['source_types']}, "
            f"broadening to all sources"
        )
        chunks = vector_query(
            query_text=question,
            ticker=ticker,
            source_types=None,
            n_results=routing["n_results"],
        )

    # Format context for the LLM
    context_str = _format_context(chunks)

    logger.info(
        f"Retrieved {len(chunks)} chunks for '{question[:50]}...' "
        f"(sources: {routing['source_types'] or 'all'})"
    )

    return {
        "context_str": context_str,
        "chunks": chunks,
        "routing": routing,
    }


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "NO CONTEXT AVAILABLE — insufficient data to answer this question."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]

        # Build citation header
        source_type = meta.get("source_type", "unknown").upper()
        doc_type = meta.get("document_type", "unknown")
        company = meta.get("company_name", meta.get("ticker", ""))
        date = meta.get("date", "")
        section = meta.get("section", "")

        # Build header line
        header_parts = [f"{source_type} {doc_type}"]
        if company:
            header_parts.append(company)
        if date:
            header_parts.append(date)
        if section:
            header_parts.append(section)

        header = " | ".join(header_parts)

        parts.append(f"[Source {i}] {header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)
