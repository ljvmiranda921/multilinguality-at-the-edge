from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_chord_diagram import chord_diagram

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_application.csv"

SECTOR_ORDER = ["Academia", "Industry", "Research\ncollective", "Government"]

SECTOR_COLORS = [
    COLORS["warm_blue"],
    COLORS["crest"],
    COLORS["warm_purple"],
    COLORS["warm_green"],
]


def build_collaboration_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build a symmetric matrix of cross-sector collaboration counts."""
    labels = ["Academia", "Industry", "Research collective", "Government"]
    n = len(labels)
    matrix = np.zeros((n, n), dtype=float)
    idx = {s: i for i, s in enumerate(labels)}

    for _, row in df.iterrows():
        types = [t.strip() for t in str(row["affiliation_types"]).split(";") if t.strip()]
        unique_types = [t for t in set(types) if t in idx]

        # Diagonal: each paper contributes once per sector it contains
        for t in unique_types:
            matrix[idx[t], idx[t]] += 1

        # Off-diagonal: one count per unique pair per paper
        for a, b in combinations(sorted(unique_types), 2):
            matrix[idx[a], idx[b]] += 1
            matrix[idx[b], idx[a]] += 1

    return matrix


def plot_chord(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    chord_diagram(
        matrix,
        ax=ax,
        names=SECTOR_ORDER,
        colors=SECTOR_COLORS,
        fontsize=18,
        use_gradient=True,
        sort="size",
        rotate_names=False,
        alpha=0.6,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "collaboration_sectors.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'collaboration_sectors.pdf'}")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers")

    matrix = build_collaboration_matrix(df)

    print("\nCollaboration matrix:")
    labels = ["Academia", "Industry", "Research collective", "Government"]
    print(f"{'':>25s}", end="")
    for s in labels:
        print(f"{s:>22s}", end="")
    print()
    for i, s in enumerate(labels):
        print(f"{s:>25s}", end="")
        for j in range(len(labels)):
            print(f"{matrix[i, j]:>22.0f}", end="")
        print()

    plot_chord(matrix)


if __name__ == "__main__":
    main()
