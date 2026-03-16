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
MANUAL_ADDITIONS = ROOT / "data" / "manual_papers.csv"

YEAR_RANGE = (2020, 2026)
MIN_RELEVANCE = 3
VALID_FOCUS = ["Efficiency", "Multilinguality", "Both"]

# Let's stagger the citation filter
CITATION_THRESHOLDS = {
    2020: 100,
    2021: 100,
    2022: 80,
    2023: 60,
    2024: 30,
    2025: 10,
    2026: 5,
}

# Don't include these after manual relevance checks
REMOVE_S2_IDS = [
    "09042877c369a9224d094c37b403735a23368316",
    "00814e5631c20bec502838a1e8040b3f5f258971",
    "00696ba295d66f049d70272219f7fea4266171be",
    "2577d053f8aab912d29b424e1f09133d83740fd2",
    "18e7ab056c16928d8f9539509a4b366889106d97",
    "93d6fa92d60938b5bd0e405e159832b91332f169",
    "fbd2c8089870814449f9254a711041bbae145a82",
    "b932a2c610a16566639f8d693eaa98181bef06f1",
    "31f44f0f2124c54e47f4df54dec63118232c25da",
    "32dcd0887537cece54e214f531d2c384470b023f",
    "455866ca838f356b53a7e3e5b344834f9e93dbbc",
    "aade40af0d85b0b4fe15c97f6222d5c2e4d6d9b3",
    "53132ea6c107479d4557631299d3ed525109b464",
    "4fcb8b6c466937025d315be6a83b624b10e860b4",
    "459c82205d2a27a8542bba7a4d478a8a23be2f5d",
    "29f032fc875576b5c3c6b1c2d76af8639bacfb88",
    "85e7d63f75c0916bd350a229e040c5fbb1472e7a",
    "9b56086e420ecb216f85d408a25264f640e46705",
]


def filter_papers(
    df: pd.DataFrame,
    year_range: tuple = YEAR_RANGE,
    min_relevance: int = MIN_RELEVANCE,
) -> pd.DataFrame:
    df = df[df["year"].between(*year_range)]
    df = df[df["relevance_score"] >= min_relevance]
    df = df[
        df.apply(
            lambda r: r["citations"] >= CITATION_THRESHOLDS.get(r["year"], 100), axis=1
        )
    ]
    df = df[df["modalities"].apply(lambda x: "Text" in ast.literal_eval(x))]
    df = df[~df["contribution_type"].str.contains("Survey")]
    df = df[~(df["contribution_type"] == "['Analysis']")]
    df = df[df["research_focus"].isin(VALID_FOCUS)]
    df = df[~df["s2_id"].isin(REMOVE_S2_IDS)]
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min_relevance", type=int, default=MIN_RELEVANCE)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n_start = len(df)
    df = filter_papers(
        df,
        min_relevance=args.min_relevance,
    )
    if MANUAL_ADDITIONS.exists():
        manual = pd.read_csv(MANUAL_ADDITIONS)
        df = pd.concat([df, manual], ignore_index=True)
        df = df.drop_duplicates(subset="s2_id", keep="last")
        print(f"Added {len(manual)} manual papers.")
    df.to_csv(args.output, index=False)
    print(f"{n_start} -> {len(df)} papers. Saved to {args.output}")


if __name__ == "__main__":
    main()
