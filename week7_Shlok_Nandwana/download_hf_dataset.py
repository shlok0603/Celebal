import argparse
import json
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

REPO_ID = "vectara/open_ragbench"
URLS_FILE = "pdf/arxiv/pdf_urls.json"


def main():
    parser = argparse.ArgumentParser(description="Download sample arXiv PDFs referenced by vectara/open_ragbench")
    parser.add_argument("--count", type=int, default=5, help="Number of PDFs to download")
    parser.add_argument("--data-dir", default="data", help="Where to place the downloaded PDFs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching the arXiv URL index from the dataset...")
    urls_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=URLS_FILE)
    with open(urls_path, "r", encoding="utf-8") as f:
        arxiv_urls = json.load(f)

    items = list(arxiv_urls.items())[: args.count]
    print(f"Downloading {len(items)} PDFs directly from arXiv...")

    for arxiv_id, url in items:
        dest = data_dir / f"{arxiv_id}.pdf"
        print(f"  {arxiv_id}  <-  {url}")
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        time.sleep(1)  # be polite to arXiv's servers

    print(f"\nDone. {len(items)} PDF(s) saved to '{data_dir}/'. Run build_index.py next.")


if __name__ == "__main__":
    main()