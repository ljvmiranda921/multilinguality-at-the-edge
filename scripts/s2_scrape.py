"""Fetches the Semantic Scholar API to get potentially relevant papers related to EdgeML."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from datasets import Dataset
from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


load_dotenv()

# Docs: https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_bulk_search
DEFAULT_BULK_QUERY = """("edge" | "mobile" | "on-device" | "tinyml" | "tiny ml" | "tiny machine learning" | "tiny deep learning") + ("language model*" | NLP | "natural language") + ("distillation" | "compression" | "quantization" | "pruning" | "efficient")"""
BULK_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk/"
# Docs: https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_relevance_search
DEFAULT_SEARCH_QUERY = "edge machine learning OR tinyML OR tiny machine learning OR microcontroller machine learning OR microcontroller neural networks OR on-device machine learning"
SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search/"
DEFAULT_FIELDS = [
    "paperId",
    "externalIds",
    "title",
    "authors",
    "abstract",
    "publicationTypes",
    "venue",
    "journal",
    "publicationDate",
    "year",
    "isOpenAccess",
    "openAccessPdf",
    "url",
    "citationCount",
    "s2FieldsOfStudy",
]


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Fetch papers from Semantic Scholar API.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output_dataset", type=str, default="ljvmiranda921/edgeml-ltl-survey", help="Path to save the fetched papers in HuggingFace Dataset format.")
    parser.add_argument("--query", required=False, default=None, help="Search query for fetching papers. If not provided, uses a default query.")
    parser.add_argument("--limit", type=int, default=-1, help="Number of papers to fetch. Set to -1 for no limit.")
    parser.add_argument("-B", "--use_bulk_api", action="store_true", help="Whether to use the bulk API for fetching papers.")
    parser.add_argument("--year", type=int, default=2015, help="Fetch papers published from this year onwards.")
    parser.add_argument("-X", "--use_unauthenticated", action="store_true", help="Use unauthenticated requests (no API key). May be subject to stricter rate limits.")
    parser.add_argument("--download", type=str, default=None, help="Directory path to download open access PDFs. If not provided, PDFs won't be downloaded.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    api_key = os.getenv("S2_API_KEY")
    if api_key is None:
        logging.error("S2_API_KEY not found in environment variables. Please set it in the .env file. See: https://www.semanticscholar.org/product/api#api-key-form")  # fmt: skip
        sys.exit(1)

    base_url = SEARCH_API if not args.use_bulk_api else BULK_API
    query = args.query
    if query is None:
        query = DEFAULT_BULK_QUERY if args.use_bulk_api else DEFAULT_SEARCH_QUERY

    papers = fetch_papers(
        query,
        base_url=base_url,
        api_key=api_key,
        year=args.year,
        limit=args.limit,
        use_unauthenticated=args.use_unauthenticated,
    )
    papers = [_cleanup(paper) for paper in papers]
    logging.info(f"Fetched {len(papers)} papers.")

    if args.download:
        download_dir = Path(args.download)
        download_dir.mkdir(parents=True, exist_ok=True)
        for paper in papers:
            _download(paper, download_dir)
        logging.info(f"Downloaded PDFs to {download_dir}")

    df = pd.DataFrame(papers).drop_duplicates(subset=["s2_id"]).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(args.output_dataset, private=True)
    logging.info(f"Saved dataset to HuggingFace Hub at {args.output_dataset}")

    output_dir = Path(f"data/{args.output_dataset.replace('/', '___')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir)
    df.to_csv(output_dir / "papers.csv", index=False)
    logging.info(f"Also saved dataset locally at {output_dir}")


def fetch_papers(
    query: str,
    *,
    base_url: str,
    api_key: str,
    year: int = 2015,
    limit: int = 100,
    use_unauthenticated: bool = False,
    fields_of_study: list[str] = ["Computer Science", "Linguistics", "Engineering"],
    publication_types: list[str] = ["JournalArticle", "Conference", "Dataset", "Study"],
) -> list[dict]:
    delay = 2.0  # seconds between requests to avoid rate limiting
    headers = {"x-api-key": api_key} if not use_unauthenticated else {}
    params = {
        "query": query,
        "year": f"{year}-",
        "fieldsOfStudy": ",".join(fields_of_study),
        "publicationTypes": ",".join(publication_types),
        "fields": ",".join(DEFAULT_FIELDS),
        "sort": "citationCount:desc",
        "minCitationCount": 5,
    }
    logging.info(f"Retrieving papers from {base_url} with query '{query}'")
    retrieved_papers = []
    while True:
        time.sleep(delay)
        logging.info(f"Querying {base_url} with the params: {params}")
        resp = requests.get(base_url, params=params, headers=headers)
        if resp.ok:
            data = resp.json()
        else:
            logging.error(f"Request failed ({resp.status_code}): {resp.text}")
            sys.exit(1)

        if "data" not in data:
            break

        retrieved_papers.extend(data.get("data"))
        logging.info(f"Retrieved {len(retrieved_papers)} papers so far (total estimate: {data.get('total', 'unknown')})")  # fmt: skip

        if limit != -1 and len(retrieved_papers) >= limit:
            logging.info(f"Reached limit of {limit} papers")
            break

        if "token" in data:
            params["token"] = data["token"]
        else:
            logging.info("No more results (no token in response)")
            break

    logging.info(f"Finished retrieving {len(retrieved_papers)} papers")
    return retrieved_papers


def _cleanup(data: dict[str, Any]) -> dict[str, Any]:
    """Attempt to flatten the data so that it's easier to view in table format"""
    paper_detail = {
        # fmt: off
        "s2_id": data.get("paperId"),
        "title": data.get("title"),
        "authors": [author.get("name") for author in data.get("authors")],
        "abstract": data.get("abstract").replace("\n", "") if data.get("abstract") else "",
        "publication_type": [pub for pub in data.get("publicationTypes")],
        "venue": data.get("venue"),
        "date": data.get("publicationDate"),
        "year": data.get("year"),
        "is_open_access": data.get("isOpenAccess"),
        "open_access_pdf": data.get("openAccessPdf").get("url") if data.get("openAccessPdf") else "",
        "s2_url": data.get("url"),
        "citations": data.get("citationCount"),
        "field_of_study": [field.get("category") for field in data.get("s2FieldsOfStudy")],
        # fmt: on
    }
    return paper_detail


def _download(paper: dict[str, Any], download_dir: Path) -> None:
    """Download the paper PDF if available"""
    pdf_url = paper.get("open_access_pdf")
    if not pdf_url:
        logging.debug(
            f"No open access PDF for paper {paper.get('s2_id')}: {paper.get('title')}"
        )
        return

    s2_id = paper.get("s2_id")
    pdf_path = download_dir / f"{s2_id}.pdf"

    if pdf_path.exists():
        logging.debug(f"PDF already exists for {s2_id}, skipping download")
        return

    try:
        logging.info(f"Downloading PDF for {s2_id}: {paper.get('title')}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with open(pdf_path, "wb") as f:
            f.write(response.content)

        logging.info(f"Successfully downloaded PDF to {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to download PDF for {s2_id}: {e}")


if __name__ == "__main__":
    main()
