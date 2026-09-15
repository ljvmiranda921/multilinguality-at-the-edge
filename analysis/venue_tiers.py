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
]

CORE_RANKED = [
    "acm on human-computer",
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
    offset = 0.22

    for side, shift in [("methods", -offset), ("deployments", offset)]:
        style = STYLE[side]
        positions = [i + shift for i in range(len(df))]
        widths = [p if n else float("nan") for p, n in zip(df[f"{side}_pct"], df[f"{side}_n"])]
        ax.barh(
            positions,
            widths,
            height=0.42,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            linewidth=1.3,
            label=f"{style['label']} (N={int(df[f'{side}_n'].sum())})",
            zorder=3,
        )
        for pos, pct, n in zip(positions, df[f"{side}_pct"], df[f"{side}_n"]):
            if n == 0:
                continue
            ax.text(
                pct + 1.2,
                pos,
                f"{pct:.1f}\\% ({n})",
                ha="left",
                va="center",
                fontsize=12,
                color=COLORS["slate_4"],
                zorder=4,
            )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    ax.set_ylim(len(df) - 0.5, -0.5)
    ax.set_xlim(0, 62)
    ax.set_xlabel("Percentage of papers (\\%)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    ax.grid(False)
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
