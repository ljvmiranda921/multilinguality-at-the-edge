from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.colors import to_rgba

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_application.csv"

DOMAIN_ORDER = [
    "Agriculture",
    "Climate",
    "Finance",
    "Healthcare",
    "Legal",
    "Social",
    "Speech",
    "General NLP",
]

DOMAIN_MAP = {
    "Crisis Response": "Social",
    "Content Moderation": "General NLP",
    "Education": "General NLP",
    "Information Retrieval": "General NLP",
    "Accessibility": "General NLP",
}

METHOD_CATEGORIES = {
    "Data Curation": ["data curation", "data filtering", "data cleaning", "corpus curation", "dataset curation"],
    "Synthetic Data": ["synthetic data", "data augmentation", "generated data", "artificial data"],
    "Tokenizer Design": ["tokenizer", "tokenization", "subword", "bpe", "sentencepiece", "vocabulary"],
    "Continual Pretraining": ["continual pretrain", "continued pretrain", "domain adaptation", "further pretrain"],
    "Parameter-Efficient": ["parameter-efficient", "peft", "lora", "adapter", "prefix tuning", "prompt tuning"],
    "Language Experts": ["mixture of experts", "moe", "language expert", "expert model", "language family expert"],
    "Model Compression": ["compression", "quantiz", "pruning", "pruned", "sparse", "int8", "int4", "gguf"],
    "Knowledge Distillation": ["distill", "student model", "teacher model", "knowledge transfer"],
    "Model Merging": ["model merging", "model fusion", "merge model", "weight averaging"],
    "Federated Learning": ["federated", "distributed learning", "privacy-preserving"],
    "Prompt Compression": ["prompt compression", "context compression", "token reduction"],
    "Speculative Decoding": ["speculative decod", "draft model", "assisted generation"],
    "Benchmark": ["benchmark", "evaluation", "test set", "leaderboard"],
}


def extract_methods(abstract: str) -> dict[str, bool]:
    abstract_lower = abstract.lower() if pd.notna(abstract) else ""
    results = {}
    for method, keywords in METHOD_CATEGORIES.items():
        results[method] = any(kw in abstract_lower for kw in keywords)
    return results


def build_cooccurrence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["domain"] = df["domain"].replace(DOMAIN_MAP)

    domains = [d for d in DOMAIN_ORDER if d in df["domain"].values]
    methods = list(METHOD_CATEGORIES.keys())

    matrix = pd.DataFrame(0, index=domains, columns=methods)

    for _, row in df.iterrows():
        domain = row["domain"]
        if domain not in domains:
            continue
        method_presence = extract_methods(row["abstract"])
        for method, present in method_presence.items():
            if present:
                matrix.loc[domain, method] = 1

    matrix = matrix.loc[:, matrix.sum() > 0]

    return matrix


def draw_bezier(ax, start, end, color, alpha=0.6, linewidth=2):
    x0, y0 = start
    x1, y1 = end
    mid_x = (x0 + x1) / 2

    verts = [(x0, y0), (mid_x, y0), (mid_x, y1), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    path = MplPath(verts, codes)

    patch = PathPatch(path, facecolor="none", edgecolor=color, alpha=alpha, linewidth=linewidth)
    ax.add_patch(patch)


def plot_bipartite(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    domains = list(matrix.index)
    methods = list(matrix.columns)

    domain_x = 0.15
    method_x = 0.85
    domain_y = np.linspace(0.9, 0.1, len(domains))
    method_y = np.linspace(0.9, 0.1, len(methods))

    domain_colors = [
        COLORS["cherry"], COLORS["crest"], COLORS["green"], COLORS["purple"],
        COLORS["indigo"], COLORS["warm_blue"], COLORS["dark_blue"], COLORS["slate_3"],
    ]

    for i, (domain, y) in enumerate(zip(domains, domain_y)):
        color = domain_colors[i % len(domain_colors)]

        for j, method in enumerate(methods):
            if matrix.loc[domain, method] == 1:
                draw_bezier(ax, (domain_x, y), (method_x, method_y[j]), color)

    for i, (domain, y) in enumerate(zip(domains, domain_y)):
        color = domain_colors[i % len(domain_colors)]
        ax.scatter(domain_x, y, s=800, c=color, zorder=3, edgecolors=COLORS["slate_4"], linewidths=1)
        ax.text(domain_x - 0.03, y, domain, ha="right", va="center", fontsize=14, fontweight="bold")

    for j, (method, y) in enumerate(zip(methods, method_y)):
        ax.scatter(method_x, y, s=600, c=COLORS["slate_2"], zorder=3, edgecolors=COLORS["slate_4"], linewidths=1, marker="s")
        ax.text(method_x + 0.03, y, method, ha="left", va="center", fontsize=12)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_method_bipartite.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'domain_method_bipartite.pdf'}")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers across {df['domain'].nunique()} domains")

    matrix = build_cooccurrence_matrix(df)
    print(f"\nDomain x Method matrix ({matrix.shape[0]} domains x {matrix.shape[1]} methods):")
    print(matrix.to_string())

    print(f"\nMethods with presence: {list(matrix.columns)}")
    print(f"Methods with no presence: {[m for m in METHOD_CATEGORIES if m not in matrix.columns]}")

    plot_bipartite(matrix)


if __name__ == "__main__":
    main()
