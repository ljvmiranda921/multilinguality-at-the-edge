"""Compute distribution of surveyed papers by category (Figure 6 stats)."""

from pathlib import Path

import pandas as pd

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

DATA_MAIN = ROOT / "data" / "papers_multilingual_edge_llm.csv"
DATA_APP = ROOT / "data" / "papers_application.csv"
DATA_BOTH = ROOT / "data" / "papers_both.csv"

CATEGORY_ORDER = ["Methodology", "Model Release", "Real-world Deployment"]


def _normalize(title: str) -> str:
    return title.strip().lower()


def main():
    df_main = pd.read_csv(DATA_MAIN)
    df_app = pd.read_csv(DATA_APP)
    df_both = pd.read_csv(DATA_BOTH)

    both_titles = set(df_both["title"].apply(_normalize))

    # Categorize main papers: model release if in both, else methodology
    records = []
    for _, row in df_main.iterrows():
        t = _normalize(row["title"])
        cat = "Model Release" if t in both_titles else "Methodology"
        records.append({"title": t, "year": int(row["year"]), "category": cat})

    # Add both papers NOT in main
    main_titles = set(df_main["title"].apply(_normalize))
    for _, row in df_both.iterrows():
        t = _normalize(row["title"])
        if t not in main_titles:
            records.append({"title": t, "year": int(row["year"]), "category": "Model Release"})

    # Add all application papers
    for _, row in df_app.iterrows():
        t = _normalize(row["title"])
        records.append({"title": t, "year": int(row["year"]), "category": "Real-world Deployment"})

    df = pd.DataFrame(records)
    total = len(df)

    print(f"Total papers: {total}")
    print()
    for cat in CATEGORY_ORDER:
        n = (df["category"] == cat).sum()
        print(f"  {cat}: {n} ({n / total * 100:.1f}%)")
    print()
    print("By year:")
    counts = df.groupby(["year", "category"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CATEGORY_ORDER, fill_value=0)
    counts["Total"] = counts.sum(axis=1)
    print(counts.to_string())


if __name__ == "__main__":
    main()
