import logging
import re
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from rank_bm25 import BM25Okapi

from src.config import CHROMA_DIR, OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ChromaDB max batch size
_BATCH_SIZE = 100

#  Singleton client

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    global _client, _collection

    if _collection is not None:
        return _collection

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    _collection = _client.get_or_create_collection(
        name="financial_docs",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    logger.info(f"ChromaDB collection 'financial_docs': {_collection.count()} documents")
    return _collection


def _make_doc_id(metadata: dict, chunk_index: int) -> str:
    parts = [
        metadata.get("ticker", "UNK"),
        metadata.get("source_type", "unk"),
        metadata.get("document_type", "unk"),
        metadata.get("date", "nodate"),
        str(chunk_index),
    ]
    return "_".join(parts).replace(" ", "_").replace("/", "_")


# ── BM25 keyword search ──────────────────────────────────────────────

_bm25_cache: dict[str, tuple] = {}  # ticker → (bm25_index, docs_list)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def _get_bm25_index(ticker: str) -> tuple[Optional[BM25Okapi], list[dict]]:
    """Build or retrieve cached BM25 index for a ticker."""
    ticker = ticker.upper()
    if ticker in _bm25_cache:
        return _bm25_cache[ticker]

    collection = _get_collection()

    # Fetch all documents for this ticker
    all_docs = collection.get(
        where={"ticker": ticker},
        include=["documents", "metadatas"],
    )

    if not all_docs["ids"]:
        return None, []

    # Tokenize and build BM25 index
    tokenized = [_tokenize(doc) for doc in all_docs["documents"]]
    bm25 = BM25Okapi(tokenized)

    docs_list = []
    for i in range(len(all_docs["ids"])):
        docs_list.append({
            "text": all_docs["documents"][i],
            "metadata": all_docs["metadatas"][i],
        })

    _bm25_cache[ticker] = (bm25, docs_list)
    logger.info(f"Built BM25 index for {ticker}: {len(docs_list)} documents")
    return bm25, docs_list


def _bm25_search(
    query_text: str,
    ticker: str,
    source_types: list[str] | None,
    n_results: int,
) -> list[dict]:
    """Run BM25 keyword search over a ticker's documents."""
    bm25, docs_list = _get_bm25_index(ticker)
    if bm25 is None:
        return []

    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    # Filter by source types and rank
    scored = []
    for i, doc in enumerate(docs_list):
        if source_types and doc["metadata"].get("source_type") not in source_types:
            continue
        scored.append((i, scores[i]))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [
        {"text": docs_list[i]["text"], "metadata": docs_list[i]["metadata"]}
        for i, _ in scored[:n_results]
    ]


def _doc_key(doc: dict) -> str:
    """Stable dedup key for a document."""
    m = doc["metadata"]
    return (
        f"{m.get('ticker', '')}_{m.get('source_type', '')}_"
        f"{m.get('document_type', '')}_{m.get('date', '')}_{m.get('chunk_index', 0)}"
    )


def _rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    n_results: int,
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion: merge two ranked lists into one."""
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys[:n_results]]


#  Public API

def add_documents(documents: list[dict]) -> int:
    if not documents:
        return 0

    collection = _get_collection()

    ids = []
    texts = []
    metadatas = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        # ChromaDB metadata values must be str, int, float, or bool
        clean_metadata = {}
        for k, v in metadata.items():
            if v is None:
                clean_metadata[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)

        doc_id = _make_doc_id(metadata, metadata.get("chunk_index", 0))
        ids.append(doc_id)
        texts.append(text)
        metadatas.append(clean_metadata)

    # Upsert in batches
    added = 0
    for i in range(0, len(ids), _BATCH_SIZE):
        batch_ids = ids[i:i + _BATCH_SIZE]
        batch_texts = texts[i:i + _BATCH_SIZE]
        batch_meta = metadatas[i:i + _BATCH_SIZE]

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        added += len(batch_ids)
        logger.debug(f"Upserted batch of {len(batch_ids)} documents")

    # Invalidate BM25 cache for affected tickers
    tickers = {m.get("ticker", "").upper() for m in metadatas}
    for t in tickers:
        _bm25_cache.pop(t, None)

    logger.info(f"Added {added} documents to ChromaDB (total: {collection.count()})")
    return added


def query(
    query_text: str,
    ticker: str | None = None,
    source_types: list[str] | None = None,
    n_results: int = 10,
) -> list[dict]:
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — no documents to search")
        return []

    # Build where filter
    where_filters = []
    if ticker:
        where_filters.append({"ticker": ticker.upper()})
    if source_types:
        where_filters.append({"source_type": {"$in": source_types}})

    where = None
    if len(where_filters) == 1:
        where = where_filters[0]
    elif len(where_filters) > 1:
        where = {"$and": where_filters}

    # ── Vector search ─────────────────────────────────────────────────
    # Fetch extra candidates for RRF merging
    vector_k = min(n_results * 2, collection.count())
    vector_results_raw = collection.query(
        query_texts=[query_text],
        n_results=vector_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    vector_results = []
    for i in range(len(vector_results_raw["ids"][0])):
        vector_results.append({
            "text": vector_results_raw["documents"][0][i],
            "metadata": vector_results_raw["metadatas"][0][i],
            "distance": vector_results_raw["distances"][0][i],
        })

    # ── BM25 keyword search ──────────────────────────────────────────
    if ticker:
        bm25_results = _bm25_search(
            query_text, ticker, source_types, n_results=n_results * 2
        )
    else:
        bm25_results = []  # BM25 index is per-ticker; skip if no ticker

    # ── Merge with Reciprocal Rank Fusion ─────────────────────────────
    if bm25_results:
        merged = _rrf_merge(vector_results, bm25_results, n_results)
        logger.info(
            f"Hybrid search: {len(vector_results)} vector + "
            f"{len(bm25_results)} BM25 → {len(merged)} merged"
        )
        return merged

    # No BM25 results (no ticker or empty index) — return vector only
    return vector_results[:n_results]


def is_ticker_indexed(ticker: str) -> bool:
    collection = _get_collection()
    ticker = ticker.upper()
    try:
        existing = collection.get(where={"ticker": ticker}, include=[], limit=1)
        return len(existing["ids"]) > 0
    except Exception:
        return False


def delete_company(ticker: str) -> int:
    collection = _get_collection()
    ticker = ticker.upper()

    # Get all IDs for this ticker
    existing = collection.get(where={"ticker": ticker}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        _bm25_cache.pop(ticker, None)
        logger.info(f"Deleted {len(existing['ids'])} documents for {ticker}")
        return len(existing["ids"])
    return 0


def get_stats() -> dict:
    collection = _get_collection()
    return {
        "total_documents": collection.count(),
        "collection_name": collection.name,
    }
