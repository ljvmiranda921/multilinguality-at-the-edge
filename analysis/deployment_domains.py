"""Extract and visualize keywords from deployment domain papers using KeyBERT."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS, get_device

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


def extract_keywords(
    df: pd.DataFrame,
    text_col: str = "description",
    top_n: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """Extract keywords from text using KeyBERT."""
    device = get_device()
    print(f"Using device: {device}")

    sentence_model = SentenceTransformer(model_name, device=device)
    kw_model = KeyBERT(model=sentence_model)

    keywords_list = []
    for _, row in df.iterrows():
        text = row[text_col]
        if pd.isna(text) or not text.strip():
            keywords_list.append([])
            continue
        # Extract keywords with diversity using MMR
        kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=top_n,
        )
        keywords_list.append([kw for kw, _ in kws])

    df = df.copy()
    df["keywords"] = keywords_list
    return df


def aggregate_keywords_by_domain(df: pd.DataFrame) -> dict[str, Counter]:
    """Aggregate keyword counts by domain."""
    domain_keywords: dict[str, Counter] = {}
    for _, row in df.iterrows():
        domain = row["domain"]
        if domain not in domain_keywords:
            domain_keywords[domain] = Counter()
        for kw in row["keywords"]:
            domain_keywords[domain][kw] += 1
    return domain_keywords


def plot_keywords_by_domain(
    domain_keywords: dict[str, Counter],
    top_n: int = 5,
) -> None:
    """Plot top keywords for each domain as horizontal bar subplots."""
    domains = [d for d in DOMAIN_ORDER if d in domain_keywords]
    n_domains = len(domains)

    fig, axes = plt.subplots(
        n_domains,
        1,
        figsize=(8, 1.5 * n_domains),
        sharex=False,
    )
    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        counter = domain_keywords[domain]
        top_kws = counter.most_common(top_n)
        if not top_kws:
            ax.set_visible(False)
            continue

        keywords = [kw for kw, _ in top_kws][::-1]
        counts = [c for _, c in top_kws][::-1]
        color = DOMAIN_STYLE.get(domain, {"color": COLORS["slate_2"]})["color"]

        ax.barh(
            keywords, counts, color=color, edgecolor=COLORS["slate_4"], linewidth=0.5
        )
        ax.set_title(domain, fontsize=16, fontweight="bold", loc="left")
        ax.set_xlim(0, max(counts) * 1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=12)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "deployment_domains_keywords.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'deployment_domains_keywords.pdf'}")


def plot_domain_distribution(df: pd.DataFrame) -> None:
    """Plot domain distribution as horizontal bar chart."""
    counts = df["domain"].value_counts()
    counts = counts.reindex([d for d in DOMAIN_ORDER if d in counts.index])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [
        DOMAIN_STYLE.get(d, {"color": COLORS["slate_2"]})["color"] for d in counts.index
    ]

    y_pos = range(len(counts))
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

    # Plot domain distribution
    plot_domain_distribution(df)

    # Extract keywords
    print("Extracting keywords with KeyBERT...")
    df = extract_keywords(df, text_col="description", top_n=5)

    # Aggregate by domain
    domain_keywords = aggregate_keywords_by_domain(df)

    # Print summary
    print("\nTop keywords per domain:")
    for domain in DOMAIN_ORDER:
        if domain in domain_keywords:
            top = domain_keywords[domain].most_common(5)
            print(f"  {domain}: {', '.join(kw for kw, _ in top)}")

    # Plot keywords
    plot_keywords_by_domain(domain_keywords, top_n=5)


if __name__ == "__main__":
    main()
