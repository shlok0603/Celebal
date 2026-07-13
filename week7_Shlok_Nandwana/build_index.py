import argparse

from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Build the RAG vector index from PDFs in a data directory.")
    parser.add_argument("--data-dir", default="data", help="Directory containing PDF files")
    parser.add_argument("--index-dir", default="index", help="Where to save the built index")
    parser.add_argument("--chunk-size", type=int, default=800, help="Characters per chunk")
    parser.add_argument("--overlap", type=int, default=150, help="Character overlap between chunks")
    args = parser.parse_args()

    pipeline = RAGPipeline(index_dir=args.index_dir)
    pipeline.build_index(data_dir=args.data_dir, chunk_size=args.chunk_size, overlap=args.overlap)


if __name__ == "__main__":
    main()
