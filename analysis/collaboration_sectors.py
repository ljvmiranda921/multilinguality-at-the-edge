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
        types = [
            t.strip() for t in str(row["affiliation_types"]).split(";") if t.strip()
        ]
        unique_types = [t for t in set(types) if t in idx]
        for t in unique_types:
            matrix[idx[t], idx[t]] += 1
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
        fontsize=24,
        use_gradient=True,
        sort="size",
        rotate_names=False,
        alpha=0.6,
        start_at=90,
    )

    # total_per_sector = matrix.sum(axis=1)
    # grand_total = total_per_sector.sum()
    # pad_deg = 2
    # total_pad = pad_deg * len(SECTOR_ORDER)
    # available = 360 - total_pad

    # order = np.argsort(-total_per_sector)
    # spans = (total_per_sector / grand_total) * available

    # angle = 90  # start_at=90
    # arc_mids = {}
    # for idx in order:
    #     span = spans[idx]
    #     mid = angle + span / 2
    #     arc_mids[idx] = mid
    #     angle += span + pad_deg

    # label_r = 1.28
    # for idx, mid_deg in arc_mids.items():
    #     a = np.radians(mid_deg)
    #     x = label_r * np.cos(a)
    #     y = label_r * np.sin(a)
    #     count = int(total_per_sector[idx])
    #     ax.text(
    #         x, y, str(count),
    #         ha="center", va="center",
    #         fontsize=20, color="black", fontweight="bold",
    #         zorder=10,
    #     )

    from matplotlib.colors import to_rgb
    from matplotlib.patches import Wedge

    for patch in ax.patches:
        if isinstance(patch, Wedge):
            fc = patch.get_facecolor()
            if hasattr(fc, "__len__") and len(fc) >= 3:
                r, g, b = to_rgb(fc[:3])
                darker = (r * 0.65, g * 0.65, b * 0.65)
                patch.set_edgecolor(darker)
                patch.set_linewidth(0.8)

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
