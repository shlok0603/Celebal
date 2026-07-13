import argparse

from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your indexed document(s).")
    parser.add_argument("--question", "-q", help="Ask a single question and exit")
    args = parser.parse_args()

    pipeline = RAGPipeline(index_dir="index")
    pipeline.load_index()

    def ask_and_print(q: str):
        result = pipeline.ask(q)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']} (page {s['page']}, score {s['score']})")

    if args.question:
        ask_and_print(args.question)
        return

    print("Document Q&A — type 'quit' to exit.\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            ask_and_print(q)


if __name__ == "__main__":
    main()
