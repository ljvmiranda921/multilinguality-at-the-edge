from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

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


def plot_heatmap(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    data = matrix.values
    ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            symbol = r"$\bullet$" if data[i, j] == 1 else r"$\circ$"
            color = COLORS["white"] if data[i, j] == 1 else COLORS["slate_2"]
            ax.text(j, i, symbol, ha="center", va="center", fontsize=20, color=color)

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(matrix.index, fontsize=14)

    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color=COLORS["slate_1"], linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_method_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'domain_method_heatmap.pdf'}")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers across {df['domain'].nunique()} domains")

    matrix = build_cooccurrence_matrix(df)
    print(f"\nDomain x Method matrix ({matrix.shape[0]} domains x {matrix.shape[1]} methods):")
    print(matrix.to_string())

    print(f"\nMethods with presence: {list(matrix.columns)}")
    print(f"Methods with no presence: {[m for m in METHOD_CATEGORIES if m not in matrix.columns]}")

    plot_heatmap(matrix)


if __name__ == "__main__":
    main()
