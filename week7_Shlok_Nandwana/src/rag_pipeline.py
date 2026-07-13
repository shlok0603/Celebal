from src.embed_store import VectorStore
from src.generator import generate_answer
from src.ingest import ingest_directory
from src.retriever import retrieve


class RAGPipeline:
    def __init__(self, index_dir: str = "index"):
        self.index_dir = index_dir
        self.store: VectorStore | None = None

    def build_index(self, data_dir: str = "data", chunk_size: int = 800, overlap: int = 150) -> None:
        """Stages 1-4: ingest documents, chunk, embed, and store."""
        chunks = ingest_directory(data_dir, chunk_size, overlap)
        self.store = VectorStore()
        self.store.build(chunks)
        self.store.save(self.index_dir)
        print(f"Index saved to '{self.index_dir}/'")

    def load_index(self) -> None:
        self.store = VectorStore.load(self.index_dir)

    def ask(self, question: str, top_k: int = 4) -> dict:
        """Stages 5-7: retrieve relevant chunks and generate an answer."""
        if self.store is None:
            self.load_index()

        retrieved = retrieve(question, self.store, top_k=top_k)
        answer = generate_answer(question, retrieved)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"source": r.chunk.source, "page": r.chunk.page, "score": round(r.score, 3)}
                for r in retrieved
            ],
        }
