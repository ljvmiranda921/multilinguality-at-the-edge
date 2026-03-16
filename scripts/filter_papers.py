"""Filter LLM-annotated papers and save a curated subset for analysis.

Reads from the full annotation CSV, applies a series of quality and relevance
filters, and writes the result to data/papers_multilingual_edge_llm.csv.
"""

import argparse
import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Defaults
DEFAULT_INPUT = ROOT / "data" / "llm_annotate" / "20260316_142727_llm_annotations.csv"
DEFAULT_OUTPUT = ROOT / "data" / "papers_multilingual_edge_llm.csv"

YEAR_RANGE = (2020, 2025)
MIN_RELEVANCE = 3
MIN_CITATIONS = 100
VALID_FOCUS = ["Efficiency", "Multilinguality", "Both"]


def filter_papers(
    df: pd.DataFrame,
    year_range: tuple = YEAR_RANGE,
    min_relevance: int = MIN_RELEVANCE,
    min_citations: int = MIN_CITATIONS,
) -> pd.DataFrame:
    n_start = len(df)
    print(f"Starting papers: {n_start}")

    df = df[df["year"].between(*year_range)]
    print(f"  After year filter ({year_range[0]}-{year_range[1]}): {len(df)}")

    df = df[df["relevance_score"] >= min_relevance]
    print(f"  After relevance >= {min_relevance}: {len(df)}")

    df = df[df["citations"] >= min_citations]
    print(f"  After citations >= {min_citations}: {len(df)}")

    df = df[df["modalities"].apply(lambda x: "Text" in ast.literal_eval(x))]
    print(f"  After text-only modality: {len(df)}")

    df = df[~df["contribution_type"].str.contains("Survey")]
    print(f"  After excluding surveys: {len(df)}")

    df = df[~(df["contribution_type"] == "['Analysis']")]
    print(f"  After excluding pure analysis: {len(df)}")

    df = df[df["research_focus"].isin(VALID_FOCUS)]
    print(f"  After keeping valid focus {VALID_FOCUS}: {len(df)}")

    print(f"\nFinal: {len(df)} papers ({len(df)/n_start:.1%} of original)")
    print(f"\nBy research focus:\n{df['research_focus'].value_counts().to_string()}")
    print(f"\nBy year:\n{df.groupby('year').size().to_string()}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min_citations", type=int, default=MIN_CITATIONS)
    parser.add_argument("--min_relevance", type=int, default=MIN_RELEVANCE)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = filter_papers(
        df,
        min_relevance=args.min_relevance,
        min_citations=args.min_citations,
    )
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
