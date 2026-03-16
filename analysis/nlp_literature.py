"""Plot publication trends in multilingual and efficient NLP literature."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_multilingual_edge_llm.csv"

FOCUS_STYLE = {
    "Efficiency": {
        "facecolor": COLORS["light_blue"],
        "edgecolor": COLORS["warm_blue"],
        "hatch": "//",
        "label": "Edge",
    },
    "Multilinguality": {
        "facecolor": COLORS["light_crest"],
        "edgecolor": COLORS["crest"],
        "hatch": "\\\\",
        "label": "Multilinguality",
    },
    "Both": {
        "facecolor": COLORS["slate_2"],
        "edgecolor": COLORS["slate_4"],
        "hatch": "",
        "label": "Both",
    },
}
FOCUS_ORDER = ["Efficiency", "Multilinguality", "Both"]

STAGE_ORDER = [
    "Data Collection",
    "Pretraining",
    "Post-training",
    "Inference",
    "Evaluation",
    "Full-Stack",
]


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers")
    df_focus = df[df["research_focus"].isin(FOCUS_ORDER)]
    df_focus = df_focus[df_focus["primary_stage"].isin(STAGE_ORDER)]

    stage_counts = (
        df_focus.groupby(["research_focus", "primary_stage"])
        .size()
        .unstack(fill_value=0)
    )
    stage_counts = stage_counts.reindex(columns=STAGE_ORDER, fill_value=0)
    stage_counts = stage_counts.reindex(FOCUS_ORDER, fill_value=0)

    # Proportional pipeline stage chart (horizontal stacked bars)
    # Reverse so Data Collection is at top
    stage_order_rev = STAGE_ORDER[::-1]
    stage_props = stage_counts.div(stage_counts.sum(axis=0), axis=1) * 100
    stage_props = stage_props.reindex(columns=stage_order_rev, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    left = np.zeros(len(stage_order_rev))
    for focus in FOCUS_ORDER:
        style = FOCUS_STYLE[focus]
        vals = stage_props.loc[focus].values
        ax.barh(
            stage_order_rev,
            vals,
            label=style["label"],
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            linewidth=1.0,
            left=left,
        )
        for j, (v, offset) in enumerate(zip(vals, left)):
            if v > 8:
                ax.text(
                    offset + v / 2,
                    j,
                    f"{v:.0f}\\%",
                    ha="center",
                    va="center",
                    fontsize=22,
                    fontweight="bold",
                )
        left += vals

    ax.set_xlabel("Percentage of papers (\\%)")
    ax.set_xlim(0, 100)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nlp_literature_by_stage_prop.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved to plot_outputs/nlp_literature_by_stage_prop.pdf")


if __name__ == "__main__":
    main()
