from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_application.csv"

DOMAIN_STYLE = {
    "Healthcare": {"color": COLORS["cherry"]},
    "Crisis Response": {"color": COLORS["crest"]},
    "Speech": {"color": COLORS["purple"]},
    "Content Moderation": {"color": COLORS["indigo"]},
    "Finance": {"color": COLORS["green"]},
    "Agriculture": {"color": COLORS["warm_green"]},
    "Education": {"color": COLORS["warm_blue"]},
    "Climate": {"color": COLORS["dark_blue"]},
    "Legal": {"color": COLORS["slate_3"]},
    "Information Retrieval": {"color": COLORS["warm_purple"]},
    "Accessibility": {"color": COLORS["warm_cherry"]},
}

DOMAIN_ORDER = [
    "Healthcare",
    "Crisis Response",
    "Speech",
    "Content Moderation",
    "Finance",
    "Agriculture",
    "Education",
    "Climate",
    "Legal",
    "Information Retrieval",
    "Accessibility",
]


def plot_domain_distribution(df: pd.DataFrame) -> None:
    counts = df["domain"].value_counts()
    counts = counts.reindex([d for d in DOMAIN_ORDER if d in counts.index])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [
        DOMAIN_STYLE.get(d, {"color": COLORS["slate_2"]})["color"] for d in counts.index
    ]

    ax.barh(
        list(counts.index)[::-1],
        list(counts.values)[::-1],
        color=colors[::-1],
        edgecolor=COLORS["slate_4"],
        linewidth=0.5,
    )

    for i, (domain, count) in enumerate(
        zip(list(counts.index)[::-1], list(counts.values)[::-1])
    ):
        ax.text(count + 0.2, i, str(count), va="center", fontsize=14)

    ax.set_xlabel("Number of papers")
    ax.set_xlim(0, counts.max() * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "deployment_domains_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'deployment_domains_distribution.pdf'}")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers across {df['domain'].nunique()} domains")
    plot_domain_distribution(df)


if __name__ == "__main__":
    main()
