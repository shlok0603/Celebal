import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import chunk_text


def test_short_text_returns_single_chunk():
    text = "Operating systems manage hardware and software resources."
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_is_split_into_multiple_chunks():
    text = "word " * 1000  # ~5000 characters
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 800


def test_chunks_overlap():
    text = "".join(f"sentence{i} " for i in range(500))
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    # the tail of one chunk should reappear at the start of the next
    assert chunks[0][-50:] in chunks[1]


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=800, overlap=150) == []


if __name__ == "__main__":
    test_short_text_returns_single_chunk()
    test_long_text_is_split_into_multiple_chunks()
    test_chunks_overlap()
    test_empty_text_returns_no_chunks()
    print("All tests passed.")
