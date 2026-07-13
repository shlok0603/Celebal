import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.ingest import Chunk

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and build a FAISS index over them."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        dim = embeddings.shape[1]
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    def save(self, out_dir: str) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_path / "index.faiss"))
        with open(out_path / "chunks.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "model_name": self.model_name}, f)

    @classmethod
    def load(cls, out_dir: str) -> "VectorStore":
        out_path = Path(out_dir)
        with open(out_path / "chunks.pkl", "rb") as f:
            data = pickle.load(f)

        store = cls(model_name=data["model_name"])
        store.chunks = data["chunks"]
        store.index = faiss.read_index(str(out_path / "index.faiss"))
        return store
