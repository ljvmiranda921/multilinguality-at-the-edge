"""Plot the venue-tier distribution of surveyed papers, split by methods vs deployments."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

MAIN_DATA_PATH = ROOT / "data" / "papers_multilingual_edge_llm.csv"
APP_DATA_PATH = ROOT / "data" / "papers_application.csv"

TIER_ORDER = [
    "Top-tier (A*)",
    "Conferences",
    "Journals",
    "Workshops",
    "Regional",
    "Preprint",
    "Grey lit.",
]

CORE_ASTAR = [
    "annual meeting of the association",
    "empirical methods",
    "neural information processing",
    "learning representations",
    "international conference on machine learning",
    "aaai conference",
    "human factors in computing",
    "acm on human-computer",
]

CORE_RANKED = [
    "north american chapter",
    "international conference on computational linguistics",
    "language resources and evaluation",
    "european chapter",
    "asia-pacific chapter",
    "interspeech",
    "artificial intelligence in education",
    "technology enhanced learning",
    "sigaccess",
    "machine translation in the americas",
    "machine translation summit",
    "computational intelligence and communication",
]

LANGUAGE_SCOPED = [
    "africanlp",
    "african natural language",
    "wanlp",
    "arabic natural language",
    "abjadnlp",
    "arabic script",
    "pacific asia",
    "nordic",
    "baltic",
    "clic-it",
    "italian conference",
    "international conference on natural language processing",
]

JOURNALS = ["nature", "plos", "ieee", "information processing & management"]
PREPRINT = ["social science research network"]
GREY = ["hugging face", "blog"]

STYLE = {
    "methods": {
        "facecolor": COLORS["light_blue"],
        "edgecolor": COLORS["warm_blue"],
        "hatch": "//",
        "label": "Methods",
    },
    "deployments": {
        "facecolor": COLORS["light_crest"],
        "edgecolor": COLORS["crest"],
        "hatch": "\\\\",
        "label": "Deployments",
    },
}

LABEL_GAP = 15.0


def classify_venue(venue: str) -> str:
    s = str(venue).strip().lower()
    if s.startswith("arxiv") or any(k in s for k in PREPRINT):
        return "Preprint"
    if any(k in s for k in GREY):
        return "Grey lit."
    if any(k in s for k in LANGUAGE_SCOPED):
        return "Regional"
    if any(k in s for k in CORE_RANKED):
        return "Conferences"
    if "workshop" in s:
        return "Workshops"
    if any(k in s for k in JOURNALS):
        return "Journals"
    if any(k in s for k in CORE_ASTAR):
        return "Top-tier (A*)"
    return "Conferences"


def load_counts() -> pd.DataFrame:
    main = pd.read_csv(MAIN_DATA_PATH)
    app = pd.read_csv(APP_DATA_PATH)

    rows = {}
    for name, df in [("methods", main), ("deployments", app)]:
        tiers = df["venue"].map(classify_venue)
        counts = tiers.value_counts().reindex(TIER_ORDER, fill_value=0)
        rows[f"{name}_n"] = counts
        rows[f"{name}_pct"] = counts / counts.sum() * 100

    return pd.DataFrame(rows).reindex(TIER_ORDER)


def plot(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    totals = df["methods_n"] + df["deployments_n"]
    labels = [f"{t} (N={int(n)})" for t, n in zip(df.index, totals)]
    baseline = df["deployments_n"].sum() / totals.sum() * 100

    left = [0.0] * len(df)
    for side in ["methods", "deployments"]:
        style = STYLE[side]
        vals = (df[f"{side}_n"] / totals * 100).to_numpy()
        ax.barh(
            labels,
            vals,
            left=left,
            height=0.72,
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            linewidth=1.0,
            label=style["label"],
        )
        for j, (v, off) in enumerate(zip(vals, left)):
            if v > 11:
                ax.text(
                    off + v / 2,
                    j,
                    f"{v:.0f}\\%",
                    ha="center",
                    va="center",
                    fontsize=17,
                    fontweight="bold",
                )
        left = [a + b for a, b in zip(left, vals)]

    ax.axvline(
        100 - baseline,
        color=COLORS["slate_3"],
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        zorder=5,
    )
    ax.text(
        100 - baseline - 1.5,
        -0.72,
        f"corpus baseline {baseline:.0f}\\%",
        ha="right",
        va="center",
        fontsize=13,
        color=COLORS["slate_3"],
    )

    ax.set_xlabel("Percentage of papers (\\%)")
    ax.set_xlim(0, 100)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_counts()
    print(df.to_string(float_format=lambda v: f"{v:.1f}"))
    outpath = OUTPUT_DIR / "venue_tiers.pdf"
    plot(df, outpath)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
