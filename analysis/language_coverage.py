"""Plot language coverage histogram for papers with reported multilingual setup."""

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils import (
    COLORS,
    OUTPUT_DIR,
    PLOT_PARAMS,
    WEB_COLORS,
    WEB_FIGURES_DIR,
    WEB_PLOT_PARAMS,
    WEB_TITLE_FONT,
)

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_multilingual_edge_llm.csv"

LANG_BINS = ["1", "2--10", "11--50", "50+"]


def _parse_langs(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def _get_num_langs(row):
    langs = row["_langs"]
    specific = [l for l in langs if l != "multilingual"]
    has_multi = "multilingual" in langs

    # Check if any "specific" entry is actually a count like '100 languages'
    extracted_count = 0
    real_codes = []
    for lang in specific:
        m = re.match(r"(\d+)\+?\s*language", lang, re.IGNORECASE)
        if m:
            extracted_count = max(extracted_count, int(m.group(1)))
        elif len(lang) <= 5 and lang.isalpha():
            real_codes.append(lang)

    if extracted_count > 0:
        return max(extracted_count, len(real_codes))

    if real_codes:
        return len(real_codes)

    if has_multi:
        text = str(row.get("title", "")) + " " + str(row.get("abstract", ""))
        matches = re.findall(r"(\d+)\+?\s*languages", text, re.IGNORECASE)
        if matches:
            return max(int(m) for m in matches)
        return -1  # unspecified multilingual

    return 0  # no info


def _bin_langs(n):
    if n == 1:
        return "1"
    elif n <= 10:
        return "2--10"
    elif n <= 50:
        return "11--50"
    else:
        return "50+"


FOCUS_ORDER = ["Efficiency", "Multilinguality", "Both"]

PAPER_FOCUS_STYLE = {
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

WEB_FOCUS_STYLE = {
    "Efficiency": {
        "facecolor": WEB_COLORS["accent_pale"],
        "edgecolor": WEB_COLORS["accent"],
        "hatch": "//",
        "label": "Edge",
    },
    "Multilinguality": {
        "facecolor": WEB_COLORS["warm_pale"],
        "edgecolor": WEB_COLORS["warm"],
        "hatch": "\\\\",
        "label": "Multilinguality",
    },
    "Both": {
        "facecolor": WEB_COLORS["cool_pale"],
        "edgecolor": WEB_COLORS["cool"],
        "hatch": "..",
        "label": "Both",
    },
}


def _plot_paper(counts, outpath):
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(LANG_BINS))
    bottom = np.zeros(len(LANG_BINS))

    for focus in FOCUS_ORDER:
        style = PAPER_FOCUS_STYLE[focus]
        vals = counts[focus].values
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=style["label"],
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            linewidth=1.0,
            width=0.6,
        )
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 3:
                ax.text(
                    x[i],
                    b + v / 2,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )
        bottom += vals

    for i, b in enumerate(LANG_BINS):
        total_val = counts.loc[b].sum()
        ax.text(
            x[i],
            total_val + 0.8,
            str(int(total_val)),
            ha="center",
            va="bottom",
            fontsize=16,
            color=COLORS["slate_3"],
        )

    ax.set_xlabel("Number of languages")
    ax.set_ylabel("Number of papers")
    ax.set_xticks(x)
    ax.set_xticklabels(LANG_BINS)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_web(counts, outpath):
    with plt.rc_context(WEB_PLOT_PARAMS):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        x = np.arange(len(LANG_BINS))
        bottom = np.zeros(len(LANG_BINS))

        for focus in FOCUS_ORDER:
            style = WEB_FOCUS_STYLE[focus]
            vals = counts[focus].values
            ax.bar(
                x,
                vals,
                bottom=bottom,
                label=style["label"],
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=1.2,
                width=0.62,
            )
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 3:
                    ax.text(
                        x[i],
                        b + v / 2,
                        str(v),
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color=style["edgecolor"],
                    )
            bottom += vals

        for i, b in enumerate(LANG_BINS):
            total_val = counts.loc[b].sum()
            ax.text(
                x[i],
                total_val + 0.8,
                str(int(total_val)),
                ha="center",
                va="bottom",
                fontsize=11,
                color=WEB_COLORS["muted"],
            )

        ax.set_xlabel("Number of languages", fontdict=WEB_TITLE_FONT)
        ax.set_ylabel("Number of papers", fontdict=WEB_TITLE_FONT)
        ax.set_xticks(x)
        ax.set_xticklabels(LANG_BINS)
        ax.legend(
            frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3
        )
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(WEB_COLORS["ink"])
        ax.spines["bottom"].set_color(WEB_COLORS["ink"])
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)


def main(export_to_web: bool = False):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers")

    df["_langs"] = df["languages_supported"].apply(_parse_langs)
    df["num_langs"] = df.apply(_get_num_langs, axis=1)

    df_known = df[df["num_langs"] > 0].copy()
    print(f"Papers with reported language setup: {len(df_known)}")

    df_known["lang_bin"] = df_known["num_langs"].apply(_bin_langs)

    bin_counts = df_known["lang_bin"].value_counts().reindex(LANG_BINS, fill_value=0)
    total = bin_counts.sum()

    print()
    for b in LANG_BINS:
        n = bin_counts[b]
        print(f"  {b}: {n} ({n / total * 100:.1f}%)")

    counts = (
        df_known.groupby(["lang_bin", "research_focus"]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(index=LANG_BINS, fill_value=0)
    counts = counts.reindex(columns=FOCUS_ORDER, fill_value=0)

    print("\nBy research focus and language bin:")
    print(counts.to_string())

    pdf_path = OUTPUT_DIR / "language_coverage.pdf"
    _plot_paper(counts, pdf_path)
    print(f"\nSaved to {pdf_path}")

    if export_to_web:
        svg_path = WEB_FIGURES_DIR / "language_coverage.svg"
        _plot_web(counts, svg_path)
        print(f"Saved to {svg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export_to_web",
        action="store_true",
        help="Also export an SVG to docs/assets/figures/.",
    )
    args = parser.parse_args()
    main(export_to_web=args.export_to_web)
