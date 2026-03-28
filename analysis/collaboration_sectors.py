from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_application.csv"

SECTOR_ORDER = ["Academia", "Industry", "Research collective", "Government"]

SECTOR_COLORS = {
    "Academia": COLORS["warm_blue"],
    "Industry": COLORS["crest"],
    "Research collective": COLORS["warm_purple"],
    "Government": COLORS["warm_green"],
}


def build_collaboration_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build a symmetric matrix of cross-sector collaboration counts."""
    n = len(SECTOR_ORDER)
    matrix = np.zeros((n, n), dtype=float)
    idx = {s: i for i, s in enumerate(SECTOR_ORDER)}

    for _, row in df.iterrows():
        types = [t.strip() for t in str(row["affiliation_types"]).split(";") if t.strip()]
        unique_types = [t for t in set(types) if t in idx]

        # Self-loops: count how many affiliations of each type
        from collections import Counter

        counts = Counter(t for t in types if t in idx)
        for t, c in counts.items():
            matrix[idx[t], idx[t]] += 1  # count once per paper per type

        # Cross-sector: one count per unique pair per paper
        for a, b in combinations(sorted(set(unique_types)), 2):
            matrix[idx[a], idx[b]] += 1
            matrix[idx[b], idx[a]] += 1

    return matrix


def plot_chord_diagram(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": False})

    n = len(SECTOR_ORDER)
    total_per_sector = matrix.sum(axis=1)
    grand_total = total_per_sector.sum()

    # Gap between sectors (in degrees)
    gap_deg = 4
    total_gap = gap_deg * n
    available = 360 - total_gap

    # Arc spans proportional to total connections
    spans = (total_per_sector / grand_total) * available

    # Compute start/end angles for each sector arc
    starts = np.zeros(n)
    ends = np.zeros(n)
    angle = 90  # start from top
    for i in range(n):
        starts[i] = angle
        ends[i] = angle - spans[i]
        angle = ends[i] - gap_deg

    # Draw outer arcs
    radius = 1.0
    arc_width = 0.08
    for i, sector in enumerate(SECTOR_ORDER):
        theta1 = ends[i]
        theta2 = starts[i]
        arc = mpatches.Arc(
            (0, 0),
            2 * radius,
            2 * radius,
            angle=0,
            theta1=theta1,
            theta2=theta2,
            linewidth=0,
        )
        # Draw as a thick wedge
        wedge = mpatches.Wedge(
            (0, 0),
            radius,
            theta1,
            theta2,
            width=arc_width,
            facecolor=SECTOR_COLORS[sector],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(wedge)

    # For each sector, subdivide its arc into slots for each connection
    # Each sector's arc is divided proportionally to its row in the matrix
    def get_arc_positions(sector_idx):
        """Return dict mapping target_idx -> (mid_angle, sub_start, sub_end)."""
        row = matrix[sector_idx]
        row_total = row.sum()
        if row_total == 0:
            return {}

        positions = {}
        s = starts[sector_idx]
        e = ends[sector_idx]
        span = s - e  # positive, going clockwise

        cur = s
        for j in range(n):
            if row[j] == 0:
                continue
            sub_span = (row[j] / row_total) * span
            sub_start = cur
            sub_end = cur - sub_span
            mid = (sub_start + sub_end) / 2
            positions[j] = (mid, sub_start, sub_end)
            cur = sub_end

        return positions

    all_positions = [get_arc_positions(i) for i in range(n)]

    # Draw chords
    inner_radius = radius - arc_width
    for i in range(n):
        for j in range(i, n):
            if matrix[i, j] == 0:
                continue

            if i == j:
                # Self-loop chord
                if i not in all_positions[i]:
                    continue
                _, s1, e1 = all_positions[i][i]
                a1 = np.radians(s1)
                a2 = np.radians(e1)
                mid_a = (a1 + a2) / 2

                p0 = (inner_radius * np.cos(a1), inner_radius * np.sin(a1))
                p3 = (inner_radius * np.cos(a2), inner_radius * np.sin(a2))
                # Control points pulled toward center
                pull = 0.3
                cp1 = (pull * np.cos(a1), pull * np.sin(a1))
                cp2 = (pull * np.cos(a2), pull * np.sin(a2))

                verts = [p0, cp1, cp2, p3]
                codes = [
                    mpath.Path.MOVETO,
                    mpath.Path.CURVE4,
                    mpath.Path.CURVE4,
                    mpath.Path.CURVE4,
                ]
                path = mpath.Path(verts, codes)
                patch = mpatches.PathPatch(
                    path,
                    facecolor=SECTOR_COLORS[SECTOR_ORDER[i]],
                    edgecolor="none",
                    alpha=0.25,
                    zorder=1,
                )
                ax.add_patch(patch)
            else:
                # Cross-sector chord
                if j not in all_positions[i] or i not in all_positions[j]:
                    continue

                _, s1, e1 = all_positions[i][j]
                _, s2, e2 = all_positions[j][i]

                a1_start = np.radians(s1)
                a1_end = np.radians(e1)
                a2_start = np.radians(s2)
                a2_end = np.radians(e2)

                # Four points on the arcs
                p0 = (inner_radius * np.cos(a1_start), inner_radius * np.sin(a1_start))
                p1 = (inner_radius * np.cos(a1_end), inner_radius * np.sin(a1_end))
                p2 = (inner_radius * np.cos(a2_start), inner_radius * np.sin(a2_start))
                p3 = (inner_radius * np.cos(a2_end), inner_radius * np.sin(a2_end))

                # Bezier control points through center
                verts = [p0, (0, 0), p2, p3, (0, 0), p1, p0]
                codes = [
                    mpath.Path.MOVETO,
                    mpath.Path.CURVE3,
                    mpath.Path.CURVE3,
                    mpath.Path.LINETO,
                    mpath.Path.CURVE3,
                    mpath.Path.CURVE3,
                    mpath.Path.CLOSEPOLY,
                ]
                path = mpath.Path(verts, codes)

                # Blend colors
                c1 = np.array(plt.cm.colors.to_rgba(SECTOR_COLORS[SECTOR_ORDER[i]]))
                c2 = np.array(plt.cm.colors.to_rgba(SECTOR_COLORS[SECTOR_ORDER[j]]))
                blend = (c1 + c2) / 2
                blend[3] = 0.35

                patch = mpatches.PathPatch(
                    path,
                    facecolor=blend,
                    edgecolor="none",
                    alpha=0.4,
                    zorder=1,
                )
                ax.add_patch(patch)

    # Labels
    label_radius = radius + 0.08
    for i, sector in enumerate(SECTOR_ORDER):
        mid_angle = (starts[i] + ends[i]) / 2
        a = np.radians(mid_angle)
        x = label_radius * np.cos(a)
        y = label_radius * np.sin(a)

        # Rotation so text follows the arc
        rot = mid_angle
        ha = "center"
        if 90 < mid_angle < 270 or -270 < mid_angle < -90:
            rot += 180
        if mid_angle % 360 > 180:
            ha = "right"
            rot = mid_angle + 180
        else:
            ha = "left"

        # Simpler: just place outside with no rotation
        ax.text(
            x * 1.12,
            y * 1.12,
            r"\textbf{" + sector + r"}",
            ha="center",
            va="center",
            fontsize=18,
            color=SECTOR_COLORS[sector],
            zorder=5,
        )

    # Count labels on arcs
    for i, sector in enumerate(SECTOR_ORDER):
        mid_angle = (starts[i] + ends[i]) / 2
        a = np.radians(mid_angle)
        x = (radius - arc_width / 2) * np.cos(a)
        y = (radius - arc_width / 2) * np.sin(a)
        count = int(total_per_sector[i])
        ax.text(
            x,
            y,
            str(count),
            ha="center",
            va="center",
            fontsize=14,
            color="white",
            fontweight="bold",
            zorder=5,
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "collaboration_sectors.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "collaboration_sectors.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'collaboration_sectors.pdf'}")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers")

    matrix = build_collaboration_matrix(df)

    print("\nCollaboration matrix:")
    print(f"{'':>25s}", end="")
    for s in SECTOR_ORDER:
        print(f"{s:>22s}", end="")
    print()
    for i, s in enumerate(SECTOR_ORDER):
        print(f"{s:>25s}", end="")
        for j in range(len(SECTOR_ORDER)):
            print(f"{matrix[i, j]:>22.0f}", end="")
        print()

    plot_chord_diagram(matrix)


if __name__ == "__main__":
    main()
