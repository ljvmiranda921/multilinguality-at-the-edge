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

# Manually flagged for removal (off-topic despite passing automated filters)
REMOVE_S2_IDS = [
    "00696ba295d66f049d70272219f7fea4266171be",  # Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space
    "2577d053f8aab912d29b424e1f09133d83740fd2",  # Multi-lingual Evaluation of Code Generation Models
    "18e7ab056c16928d8f9539509a4b366889106d97",  # StarCoder 2 and The Stack v2: The Next Generation
    "93d6fa92d60938b5bd0e405e159832b91332f169",  # Is ChatGPT a Highly Fluent Grammatical Error Correction System?
    "fbd2c8089870814449f9254a711041bbae145a82",  # How Far Can Camels Go?
    "b932a2c610a16566639f8d693eaa98181bef06f1",  # With More Contexts Comes Better Performance
    "31f44f0f2124c54e47f4df54dec63118232c25da",  # ChatGPT: Beginning of an End of Manual Linguistic Data Annotation?
    "32dcd0887537cece54e214f531d2c384470b023f",  # Large Language Models as Tool Makers
    "455866ca838f356b53a7e3e5b344834f9e93dbbc",  # ToolAlpaca
    "aade40af0d85b0b4fe15c97f6222d5c2e4d6d9b3",  # Graph of Thoughts
    "53132ea6c107479d4557631299d3ed525109b464",  # AFlow: Automating Agentic Workflow Generation
    "4fcb8b6c466937025d315be6a83b624b10e860b4",  # MAmmoTH2: Scaling Instructions from the Web
    "459c82205d2a27a8542bba7a4d478a8a23be2f5d",  # Is ChatGPT Good at Search?
    "29f032fc875576b5c3c6b1c2d76af8639bacfb88",  # OpenChat
    "85e7d63f75c0916bd350a229e040c5fbb1472e7a",  # Making Pre-trained Language Models Better Few-shot Learners
    "9b56086e420ecb216f85d408a25264f640e46705",  # Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners
]


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

    df = df[~df["s2_id"].isin(REMOVE_S2_IDS)]
    print(f"  After manual removal list ({len(REMOVE_S2_IDS)} IDs): {len(df)}")

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
