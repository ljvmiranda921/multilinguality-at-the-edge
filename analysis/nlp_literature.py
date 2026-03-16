"""Plot publication trends in multilingual and efficient NLP literature."""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils import COLORS, PLOT_PARAMS, OUTPUT_DIR

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "llm_annotate" / "20260316_142727_llm_annotations copy.csv"

FOCUS_STYLE = {
    "Efficiency": {"color": COLORS["indigo"], "label": "Efficiency"},
    "Multilinguality": {"color": COLORS["cherry"], "label": "Multilinguality"},
    "Both": {"color": COLORS["dark_green"], "label": "Both"},
}
FOCUS_ORDER = ["Efficiency", "Multilinguality", "Both"]

# Filters
YEAR_RANGE = (2020, 2025)
MIN_RELEVANCE = 3
MIN_CITATIONS = 40
REQUIRE_TEXT_MODALITY = True
EXCLUDE_SURVEYS = True


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard filters to the annotation dataframe."""
    df = df[df["year"].between(*YEAR_RANGE)]
    df = df[df["relevance_score"] >= MIN_RELEVANCE]
    df = df[df["citations"] >= MIN_CITATIONS]
    if REQUIRE_TEXT_MODALITY:
        df = df[df["modalities"].apply(lambda x: "Text" in ast.literal_eval(x))]
    if EXCLUDE_SURVEYS:
        df = df[~df["contribution_type"].str.contains("Survey")]
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = _apply_filters(df)
    print(f"After filtering: {len(df)} papers")
    df_focus = df[df["research_focus"].isin(FOCUS_ORDER)]

    counts = df_focus.groupby(["year", "research_focus"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=FOCUS_ORDER, fill_value=0)
    years = counts.index.tolist()

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(years))
    width = 0.25

    for i, focus in enumerate(FOCUS_ORDER):
        style = FOCUS_STYLE[focus]
        ax.bar(
            x + i * width,
            counts[focus],
            width,
            label=style["label"],
            color=style["color"],
            edgecolor=COLORS["slate_3"],
            linewidth=0.5,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_xticks(x + width)
    ax.set_xticklabels(years)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nlp_literature_by_focus.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved to plot_outputs/nlp_literature_by_focus.pdf")

    proportions = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(years))
    for focus in FOCUS_ORDER:
        style = FOCUS_STYLE[focus]
        ax.bar(
            years,
            proportions[focus],
            label=style["label"],
            color=style["color"],
            edgecolor=COLORS["slate_3"],
            linewidth=0.5,
            bottom=bottom,
        )
        bottom += proportions[focus].values

    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of papers")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nlp_literature_by_focus_prop.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved to plot_outputs/nlp_literature_by_focus_prop.pdf")

    df_focus = df_focus.copy()
    df_focus["pipeline_stages"] = df_focus["pipeline_stages"].apply(ast.literal_eval)
    stages_exploded = df_focus.explode("pipeline_stages")
    stage_counts = (
        stages_exploded.groupby(["research_focus", "pipeline_stages"])
        .size()
        .unstack(fill_value=0)
    )
    stage_order = [
        "Data Collection",
        "Pretraining",
        "Post-training",
        "Inference",
        "Evaluation",
    ]
    stage_counts = stage_counts.reindex(columns=stage_order, fill_value=0)
    stage_counts = stage_counts.reindex(FOCUS_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stage_order))
    width = 0.25

    for i, focus in enumerate(FOCUS_ORDER):
        style = FOCUS_STYLE[focus]
        ax.bar(
            x + i * width,
            stage_counts.loc[focus],
            width,
            label=style["label"],
            color=style["color"],
            edgecolor=COLORS["slate_3"],
            linewidth=0.5,
        )

    ax.set_xlabel("Pipeline stage")
    ax.set_ylabel("Number of papers")
    ax.set_xticks(x + width)
    ax.set_xticklabels(stage_order, rotation=15, ha="right")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nlp_literature_by_stage.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved to plot_outputs/nlp_literature_by_stage.pdf")


if __name__ == "__main__":
    main()
