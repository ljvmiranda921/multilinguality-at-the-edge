"""Plot deployment mode against hardware class for deployed edge LM systems."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS, save_paper_figure

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_application.csv"

HARDWARE_ORDER = [
    "Microcontrollers",
    "Single-board Computers",
    "Smartphones",
    "Consumer PCs",
]

HARDWARE_LABELS = {
    "Microcontrollers": "Microcontrollers",
    "Single-board Computers": "Single-board",
    "Smartphones": "Smartphones",
    "Consumer PCs": "Consumer PCs",
}

MODE_ORDER = ["on-device only", "both", "client only"]

MODE_STYLE = {
    "on-device only": {
        "facecolor": COLORS["light_blue"],
        "edgecolor": COLORS["warm_blue"],
        "hatch": "//",
        "label": "On-device",
    },
    "both": {
        "facecolor": COLORS["slate_1"],
        "edgecolor": COLORS["slate_3"],
        "hatch": "",
        "label": "Both",
    },
    "client only": {
        "facecolor": COLORS["light_crest"],
        "edgecolor": COLORS["crest"],
        "hatch": "\\\\",
        "label": "Client (API)",
    },
}


def split_field(value: str) -> list[str]:
    return [p.strip() for p in str(value).split(";") if p.strip() and p.strip() != "N/A"]


def categorize(modes: list[str]) -> str | None:
    local = bool({"on-device", "server"} & set(modes))
    remote = "client" in modes
    if local and remote:
        return "both"
    if local:
        return "on-device only"
    if remote:
        return "client only"
    return None


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    counts = pd.DataFrame(0, index=HARDWARE_ORDER, columns=MODE_ORDER, dtype=float)
    papers = pd.Series(0, index=HARDWARE_ORDER, dtype=int)

    for _, row in df.iterrows():
        classes = [c for c in split_field(row["hardware_class"]) if c in HARDWARE_ORDER]
        category = categorize(split_field(row["deployment_mode"]))
        if not classes or category is None:
            continue
        for cls in classes:
            papers[cls] += 1
            counts.loc[cls, category] += 1

    return counts, papers


def plot(counts: pd.DataFrame, papers: pd.Series, outpath: Path) -> None:
    props = counts.div(counts.sum(axis=1), axis=0) * 100
    order = HARDWARE_ORDER[::-1]
    props = props.reindex(order)

    labels = [f"{HARDWARE_LABELS[h]} (N={int(papers[h])})" for h in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    left = np.zeros(len(order))
    for mode in MODE_ORDER:
        style = MODE_STYLE[mode]
        vals = props[mode].to_numpy()
        ax.barh(
            labels,
            vals,
            left=left,
            height=0.66,
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
        left += vals

    ax.set_xlabel("Percentage of deployments (\\%)")
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
    save_paper_figure(fig, outpath)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    counts, papers = build_matrix(df)

    print(counts.astype(int).to_string())
    print()
    print(f"papers per class:\n{papers.to_string()}")
    covered = int(papers.sum())
    print(f"\ncoverage: {covered} class-assignments from {len(df)} deployment papers")

    outpath = OUTPUT_DIR / "deployment_hardware.pdf"
    plot(counts, papers, outpath)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
