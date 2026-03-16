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
    2020: 200,
    2021: 200,
    2022: 150,
    2023: 100,
    2024: 80,
    2025: 20,
    2026: 5,
}

# Don't include these after manual relevance checks
REMOVE_S2_IDS = [
    "0d5124d1fb7f21aa2efc0ae234feab97e8a23208",
    "0c9d97d2ba489256d4f1760598dc2c7be6d90d96",
    "0a168c38959a8bdb5f5cf8b91b06e64e3a77f27a",
    "09dff683262090feaa9b7b97c1c43103cc96657d",
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
    "0edc2883c836aad28e559d4a4ceff805dbe808bb",
    "1067c44e473b6998f89e13f0d4c0de730def43f0",
    "111359fbc0fe744f969cb0f1b66ae3e2bf4e8685",
    "13b8934468665ecb586f491d7f9f6c460cb095e5",
    "15c49cb2a3d422e757b990fa2bbc327539cd4db7",
    "164e9a9ab16ecf675d1a82325282ef0aaa29d3b7",
    "165dca9b1794ae105ba4fc5c984971deedd33bd4",
    "188336f606e76fda9e219b954d1750ad26646fdb",
    "1959083b64a50b8582aa00452011dbed3e95331a",
    "1aa1d6b29ad6fcef78d1eefacb2a7fd75e68c2c0",
    "1bf21dabbdfc81fd4f9e92b1201ecce744cabb6a",
    "1caf450f88031258484dcb616daba512fdf78774",
    "1e6bce5cb89e60662fa597d1fb46ad1e16176399",
    "23af54b82c951317f1fc1841164d8a441a2d8120",
    "23c2e21b6a5789724715f2986fdc586a517ffe57",
    "28660d983370b06442bf6cc856327f3278f53599",
    "fec9e5449c5a03792ee7fc1875e57a6efcb97e1e",
    "fdb10db0ce04e42f4aad52048991d8ccfaad2a4d",
    "fbfa1d0ec0074478311196fc667cb7bae276c8f1",
    "fb0e6743bc71d455ee3857e5f8fe6488b0ee89ff",
    "fb2307f7ce7c6868429ee3ee15d6eaf311ecba5c",
    "fa64368a86b9cd8077ced70eb8d46785f3baa0be",
    "faab24bc6cd4a4dea6e82420d145f08445c05fc7",
    "f7c14e79d3eb1d24b4184d106244be1672113ce2",
    "f69f494ab691481ec353da4be480b334fada6607",
    "f608011b0f50a14bb2949c186a7c632a099aa75b",
    "f1d477ccd20b3e90611fc46b1951b3708651a425",
    # Second round of manual review
    "26f82f42f434fdf190e0b3004601ee93ae6ace9e",  # Rank1: IR reranking
    "40d8e46e117013f3e3143bb2ad03b3fb5c58f00b",  # Fraud-R1: fraud benchmark
    "cde48b24264e44355abb0e548a2cf7c70bb072b4",  # NV-Embed: embedding model
    "c491d6e6e24db88e1532788d63bbecfc02b8d9a0",  # Serving LLMs on Huawei CloudMatrix384
    "d0e7353189feb4501c637b7008ff993de603c3f0",  # Nemotron-4 340B: huge model
    "c4022c8b34254068af24ca00b5b7d5aa250ce2bf",  # Llama2Vec: dense retrieval
    "ae4aa7b0c5a327dae63ad20e00b58bcc6176b3b8",  # Moshi: speech foundation model
    "ad2be51acf42f686a8d1de92d7435d84274ee62d",  # Orca-Math: math reasoning
    "f8d75c913a4811fe9f2030ca8d965c3a6d49b423",  # GraphRouter: LLM routing
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
