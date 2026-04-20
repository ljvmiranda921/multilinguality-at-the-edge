"""Scrape Semantic Scholar for multilingual and efficient NLP papers."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

load_dotenv()

BULK_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk/"
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

# We run multiple queries to capture both fields and their intersection.
# Each query is a (name, bulk_query) tuple.
QUERIES = {
    "multilingual": (
        '("multilingual" | "cross-lingual" | "crosslingual" | "low-resource language*")'
        " + "
        '("language model*" | "NLP" | "natural language processing"'
        ' | "LLM" | "large language model*")'
    ),
    "efficient": (
        '("compression" | "quantization" | "pruning" | "distillation"'
        ' | "edge" | "on-device" | "tinyml" | "lightweight"'
        ' | "small language model*" | "knowledge distillation")'
        " + "
        '("language model*" | "NLP" | "natural language processing"'
        ' | "LLM" | "large language model*")'
    ),
    "intersection": (
        '("multilingual" | "cross-lingual" | "crosslingual" | "low-resource language*")'
        " + "
        '("compression" | "quantization" | "pruning" | "distillation"'
        ' | "edge" | "on-device" | "tinyml" | "lightweight"'
        ' | "small language model*" | "knowledge distillation")'
        " + "
        '("language model*" | "NLP" | "natural language processing"'
        ' | "LLM" | "large language model*")'
    ),
}

# fmt: off
DEFAULT_VENUES = [
    "Annual Meeting of the Association for Computational Linguistics",
    "Conference on Empirical Methods in Natural Language Processing",
    "North American Chapter of the Association for Computational Linguistics",
    "Conference of the European Chapter of the Association for Computational Linguistics",
    "International Conference on Computational Linguistics",
    "International Conference on Learning Representations",
    "Neural Information Processing Systems",
    "International Conference on Machine Learning",
    "AAAI Conference on Artificial Intelligence",
    "arXiv.org",
    "Trans. Mach. Learn. Res.",
]
# fmt: on


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Fetch papers from Semantic Scholar API.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output_dir", type=str, default="data/s2_papers", help="Directory to save CSV output.")
    parser.add_argument("--query_names", nargs="*", default=None, help="Which queries to run (multilingual, efficient, intersection). Default: all.")
    parser.add_argument("--limit", type=int, default=-1, help="Number of papers to fetch per query. Set to -1 for no limit.")
    parser.add_argument("--year", type=int, default=2020, help="Fetch papers published from this year onwards.")
    parser.add_argument("--min_citations", type=int, default=5, help="Minimum citation count filter.")
    parser.add_argument("--download", type=str, default=None, help="Directory path to download open access PDFs.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    api_key = os.getenv("S2_API_KEY")
    if api_key is None:
        logging.warning(
            "S2_API_KEY not found. Using unauthenticated requests (stricter rate limits)."
        )

    query_names = args.query_names or list(QUERIES.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_papers = []
    for name in query_names:
        if name not in QUERIES:
            logging.warning(f"Unknown query '{name}', skipping.")
            continue
        query = QUERIES[name]
        logging.info(f"Running query: {name}")
        papers = fetch_papers(
            query,
            api_key=api_key,
            year=args.year,
            limit=args.limit,
            min_citations=args.min_citations,
            venues=DEFAULT_VENUES,
        )
        for p in papers:
            p["query_source"] = name
        all_papers.extend(papers)
        logging.info(f"  -> {len(papers)} papers for '{name}'")

    papers = [_cleanup(p) for p in all_papers]
    logging.info(f"Total papers fetched: {len(papers)}")

    df = pd.DataFrame(papers)
    # Deduplicate: keep all query_source tags for a paper
    df = df.groupby("s2_id", as_index=False).agg(
        {
            **{
                col: "first"
                for col in df.columns
                if col not in ("s2_id", "query_source")
            },
            "query_source": lambda x: ",".join(sorted(set(x))),
        }
    )
    logging.info(f"After dedup: {len(df)} unique papers")

    if args.download:
        download_dir = Path(args.download)
        download_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            _download(row.to_dict(), download_dir)

    # Save
    df.to_csv(output_dir / "papers.csv", index=False)
    logging.info(f"Saved {len(df)} papers to {output_dir / 'papers.csv'}")

    for name in query_names:
        subset = df[df["query_source"].str.contains(name)]
        subset.to_csv(output_dir / f"papers_{name}.csv", index=False)
        logging.info(
            f"  -> {len(subset)} papers in {output_dir / f'papers_{name}.csv'}"
        )


def fetch_papers(
    query: str,
    *,
    api_key: str | None = None,
    year: int = 2015,
    limit: int = -1,
    min_citations: int = 5,
    fields_of_study: list[str] = ["Computer Science", "Linguistics"],
    publication_types: list[str] = ["JournalArticle", "Conference"],
    venues: list[str] | None = None,
) -> list[dict]:
    delay = 2.0
    headers = {"x-api-key": api_key} if api_key else {}
    params = {
        "query": query,
        "year": f"{year}-",
        "fieldsOfStudy": ",".join(fields_of_study),
        "publicationTypes": ",".join(publication_types),
        "fields": ",".join(DEFAULT_FIELDS),
        "sort": "citationCount:desc",
        "minCitationCount": min_citations,
    }
    if venues:
        params["venue"] = ",".join(venues)
    logging.info(f"Retrieving papers with query: {query[:100]}...")
    retrieved_papers = []
    while True:
        time.sleep(delay)
        resp = requests.get(BULK_API, params=params, headers=headers)
        if resp.ok:
            data = resp.json()
        else:
            logging.error(f"Request failed ({resp.status_code}): {resp.text}")
            sys.exit(1)

        if "data" not in data:
            break

        batch = data["data"]
        if not batch:
            logging.info("Empty batch, stopping")
            break

        retrieved_papers.extend(batch)
        total = data.get("total", "?")
        logging.info(
            f"Retrieved {len(retrieved_papers)} papers so far (total: {total})"
        )

        if isinstance(total, int) and len(retrieved_papers) >= total:
            logging.info("Fetched all available papers")
            break

        if limit != -1 and len(retrieved_papers) >= limit:
            logging.info(f"Reached limit of {limit} papers")
            break

        if "token" in data:
            params["token"] = data["token"]
        else:
            logging.info("No more results")
            break

    return retrieved_papers


def _cleanup(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten paper data into a tabular format."""
    return {
        # fmt: off
        "s2_id": data.get("paperId"),
        "title": data.get("title"),
        "authors": [author.get("name") for author in data.get("authors", [])],
        "abstract": data.get("abstract", "").replace("\n", "") if data.get("abstract") else "",
        "publication_type": [pub for pub in data.get("publicationTypes", [])],
        "venue": data.get("venue"),
        "date": data.get("publicationDate"),
        "year": data.get("year"),
        "is_open_access": data.get("isOpenAccess"),
        "open_access_pdf": data.get("openAccessPdf", {}).get("url", "") if data.get("openAccessPdf") else "",
        "s2_url": data.get("url"),
        "citations": data.get("citationCount"),
        "field_of_study": [field.get("category") for field in data.get("s2FieldsOfStudy", [])],
        "query_source": data.get("query_source", ""),
        # fmt: on
    }


def _download(paper: dict[str, Any], download_dir: Path) -> None:
    """Download the paper PDF if available."""
    pdf_url = paper.get("open_access_pdf")
    if not pdf_url:
        return

    s2_id = paper.get("s2_id")
    pdf_path = download_dir / f"{s2_id}.pdf"
    if pdf_path.exists():
        return

    try:
        logging.info(f"Downloading PDF for {s2_id}: {paper.get('title')}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        logging.error(f"Failed to download PDF for {s2_id}: {e}")


if __name__ == "__main__":
    main()
