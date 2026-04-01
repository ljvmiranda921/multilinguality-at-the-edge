"""Plot model size ranges per model family from papers_both.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_both.csv"

NAME_MAP = {
    "EmbeddingGemma: Powerful and Lightweight Text Representations": "EmbeddingGemma",
    "Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance": "Falcon-H1",
    "GLM-130B: An Open Bilingual Pre-trained Model": "GLM-130B",
    "Mixtral of Experts": "Mixtral 8x7B",
    "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models": "Qwen3 Embedding",
    "Qwen2 Technical Report": "Qwen2",
    "SeaLLMs - Large Language Models for Southeast Asia": "SeaLLMs",
    "Qwen2.5 Technical Report": "Qwen2.5",
    "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone": "Phi-3",
    "PaLM 2 Technical Report": "PaLM 2",
    "Yi: Open Foundation Models by 01.AI": "Yi",
    "Tiny Aya: Bridging Scale and Multilingual Depth": "Tiny Aya",
    "Gemma 3 Technical Report": "Gemma 3",
    "TranslateGemma Technical Report": "TranslateGemma",
    "No Language Left Behind: Scaling Human-Centered Machine Translation": "NLLB",
    "The Llama 3 Herd of Models": "Llama 3",
    "EuroLLM: Multilingual Language Models for Europe": "EuroLLM",
    "MaLA-500: Massive Language Adaptation of Large Language Models": "MaLA-500",
    "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs": "Phi-4-Mini",
    "NVIDIA Nemotron 3: Efficient and Open Intelligence": "Nemotron 3",
    "Granite 3.0 Language Models": "Granite 3.0",
    "Compact Language Models via Pruning and Knowledge Distillation": "Minitron",
    "Orca: Progressive Learning from Complex Explanation Traces of GPT-4": "Orca",
    "Orca 2: Teaching Small Language Models How to Reason": "Orca 2",
    "Textbooks Are All You Need": "Phi-1",
    "Textbooks Are All You Need II: phi-1.5 Technical Report": "Phi-1.5",
    "Phi-2: The Surprising Power of Small Language Models": "Phi-2",
    "Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages": "Omnilingual ASR",
}


def main():
    df = pd.read_csv(DATA_PATH)

    records = []
    for _, row in df.iterrows():
        s = str(row["model_size"])
        if s in ("not specified", "nan"):
            continue
        sizes = []
        for val in s.split(";"):
            try:
                sizes.append(float(val))
            except ValueError:
                continue
        if sizes:
            name = NAME_MAP.get(row["title"], row["title"][:30])
            records.append(
                {
                    "name": name,
                    "sizes": sorted(sizes),
                    "min": min(sizes),
                    "max": max(sizes),
                    "year": int(row["year"]),
                }
            )

    records.sort(key=lambda r: (r["year"], r["min"]))

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    y_positions = np.arange(len(records))

    for i, rec in enumerate(records):
        sizes = rec["sizes"]
        if len(sizes) == 1:
            ax.plot(
                sizes[0],
                i,
                "o",
                color=COLORS["crest"],
                markersize=10,
                markeredgecolor=COLORS["dark_crest"],
                markeredgewidth=1.0,
                zorder=3,
            )
        else:
            ax.plot(
                [sizes[0], sizes[-1]],
                [i, i],
                color=COLORS["warm_blue"],
                linewidth=2.5,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                sizes,
                [i] * len(sizes),
                "o",
                color=COLORS["crest"],
                markersize=8,
                markeredgecolor=COLORS["dark_crest"],
                markeredgewidth=1.0,
                zorder=3,
            )

    years = [r["year"] for r in records]
    prev_year = None
    for i, yr in enumerate(years):
        if prev_year is not None and yr != prev_year:
            ax.axhline(i - 0.5, color=COLORS["slate_2"], linewidth=0.8, linestyle="-")
        prev_year = yr

    year_groups = {}
    for i, yr in enumerate(years):
        year_groups.setdefault(yr, []).append(i)
    for yr, indices in year_groups.items():
        mid = np.mean(indices)
        ax.text(
            620,
            mid,
            str(yr),
            ha="left",
            va="center",
            fontsize=16,
            color=COLORS["slate_3"],
            style="italic",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["name"] for r in records])
    ax.set_xscale("log")
    ax.set_xlabel("Model size (B parameters)")

    tick_vals = [0.3, 1, 3, 7, 14, 30, 70, 130, 400]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([str(v) for v in tick_vals])
    ax.set_xlim(0.2, 600)

    ax.axvspan(0.2, 8, alpha=0.10, color=COLORS["cambridge_blue"], zorder=0)
    ax.axvspan(8, 80, alpha=0.10, color=COLORS["judge_yellow"], zorder=0)
    ax.axvspan(80, 600, alpha=0.10, color=COLORS["dark_crest"], zorder=0)

    label_y = len(records) - 0.1
    ax.text(
        1.3,
        label_y,
        r"Small",
        ha="center",
        va="bottom",
        fontsize=22,
        color=COLORS["dark_blue"],
    )
    ax.text(
        25,
        label_y,
        r"Medium",
        ha="center",
        va="bottom",
        fontsize=22,
        color=COLORS["slate_4"],
    )
    ax.text(
        220,
        label_y,
        r"Large",
        ha="center",
        va="bottom",
        fontsize=22,
        color=COLORS["dark_crest"],
    )

    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    outpath = OUTPUT_DIR / "model_sizes.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
