from pathlib import Path
from dataclasses import dataclass
import fitz 


@dataclass
class Chunk:
    text: str
    source: str      # filename the chunk came from
    page: int        # 1-indexed page number
    chunk_id: int     # index of this chunk within the document


def load_pdf_text(pdf_path: str) -> list[tuple[int, str]]:
    """Load a PDF and return a list of (page_number, page_text)."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append((i, text))
    doc.close()
    return pages

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    chunk_size / overlap are measured in characters. Overlap keeps context
    from being cut off awkwardly at chunk boundaries, which improves
    retrieval accuracy (mentioned as an improvement idea in the assignment).
    """
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def ingest_pdf(pdf_path: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Full ingestion for a single PDF: load -> chunk -> tag with metadata."""
    source = Path(pdf_path).name
    pages = load_pdf_text(pdf_path)

    all_chunks: list[Chunk] = []
    running_id = 0
    for page_num, page_text in pages:
        for piece in chunk_text(page_text, chunk_size, overlap):
            all_chunks.append(Chunk(text=piece, source=source, page=page_num, chunk_id=running_id))
            running_id += 1
    return all_chunks


def ingest_directory(data_dir: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Ingest every PDF found in a directory."""
    data_path = Path(data_dir)
    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in '{data_dir}'. Add a PDF (see data/README.md) and try again."
        )

    all_chunks: list[Chunk] = []
    for pdf_file in pdf_files:
        print(f"Ingesting {pdf_file.name} ...")
        all_chunks.extend(ingest_pdf(str(pdf_file), chunk_size, overlap))
    print(f"Done. {len(all_chunks)} chunks created from {len(pdf_files)} file(s).")
    return all_chunks
