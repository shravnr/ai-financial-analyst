import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.config import CHROMA_DIR, OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ChromaDB max batch size
_BATCH_SIZE = 100

# ── Singleton client ──────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    """Get or create the financial_docs collection with OpenAI embeddings."""
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


# ── Public API ────────────────────────────────────────────────────────

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

    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten results (query returns nested lists)
    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return docs


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
        logger.info(f"Deleted {len(existing['ids'])} documents for {ticker}")
        return len(existing["ids"])
    return 0


def get_stats() -> dict:
    collection = _get_collection()
    return {
        "total_documents": collection.count(),
        "collection_name": collection.name,
    }
