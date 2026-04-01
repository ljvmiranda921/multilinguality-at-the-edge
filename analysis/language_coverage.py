"""Plot language coverage histogram for papers with reported multilingual setup."""

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_multilingual_edge_llm.csv"

LANG_BINS = ["1", "2--10", "11--50", "50+"]


def _parse_langs(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def _get_num_langs(row):
    langs = row["_langs"]
    specific = [l for l in langs if l != "multilingual"]
    has_multi = "multilingual" in langs

    if specific:
        return len(specific)

    if has_multi:
        text = str(row.get("title", "")) + " " + str(row.get("abstract", ""))
        matches = re.findall(r"(\d+)\+?\s*languages", text, re.IGNORECASE)
        if matches:
            return max(int(m) for m in matches)
        return -1  # unspecified multilingual

    return 0  # no info


def _bin_langs(n):
    if n == 1:
        return "1"
    elif n <= 10:
        return "2--10"
    elif n <= 50:
        return "11--50"
    else:
        return "50+"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers")

    df["_langs"] = df["languages_supported"].apply(_parse_langs)
    df["num_langs"] = df.apply(_get_num_langs, axis=1)

    # Keep only papers with a reported language count (> 0)
    df_known = df[df["num_langs"] > 0].copy()
    print(f"Papers with reported language setup: {len(df_known)}")

    df_known["lang_bin"] = df_known["num_langs"].apply(_bin_langs)

    # Count per bin
    bin_counts = df_known["lang_bin"].value_counts().reindex(LANG_BINS, fill_value=0)
    total = bin_counts.sum()

    print()
    for b in LANG_BINS:
        n = bin_counts[b]
        print(f"  {b}: {n} ({n / total * 100:.1f}%)")

    # Histogram-style bar chart
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(LANG_BINS))
    bars = ax.bar(
        x,
        bin_counts.values,
        color=COLORS["cambridge_blue"],
        edgecolor=COLORS["dark_blue"],
        linewidth=1.0,
        width=0.6,
    )

    # Add count labels on top of each bar
    for i, (bar, val) in enumerate(zip(bars, bin_counts.values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val),
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
        )

    ax.set_xlabel("Number of languages")
    ax.set_ylabel("Number of papers")
    ax.set_xticks(x)
    ax.set_xticklabels(LANG_BINS)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    outpath = OUTPUT_DIR / "language_coverage.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
