from dataclasses import dataclass

from src.embed_store import VectorStore
from src.ingest import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def retrieve(query: str, store: VectorStore, top_k: int = 4) -> list[RetrievedChunk]:
    """Return the top_k most relevant chunks for a query, ranked by similarity."""
    query_vec = store.embed_query(query)
    scores, indices = store.index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append(RetrievedChunk(chunk=store.chunks[idx], score=float(score)))
    return results
