from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

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
]

DOMAIN_MAP = {
    "Crisis Response": "Social",
    "Content Moderation": "Social",
    "Education": "Social",
    "Information Retrieval": "Social",
    "Accessibility": "Social",
}

DOMAIN_COLORS = {
    "Agriculture": COLORS["warm_green"],
    "Climate": COLORS["dark_blue"],
    "Finance": COLORS["green"],
    "Healthcare": COLORS["cherry"],
    "Legal": COLORS["slate_3"],
    "Social": COLORS["crest"],
    "Speech": COLORS["purple"],
}

METHOD_KEYWORDS = {
    "RAG": ["rag", "retrieval-augmented", "retrieval augmented", "retrieval"],
    "LoRA": ["lora", "qlora", "adapter", "peft"],
    "Quantization": ["quantiz", "int8", "int4", "ptq", "gguf", "bit precision"],
    "Distillation": ["distil", "student model", "teacher model"],
    "Fine-tuning": ["fine-tun", "finetuning", "instruction tuning", "sft"],
    "MoE": ["mixture of experts", "moe", "language family expert", "expert model"],
    "Federated": ["federated", "privacy-preserving", "distributed learning"],
    "Synthetic Data": ["synthetic data", "data augmentation", "generated data", "artificial data"],
    "Continual Pretraining": ["continual pretrain", "continued pretrain", "further pretrain", "domain adaptation"],
    "On-device": ["on-device", "edge deploy", "mobile", "lightweight", "on device"],
    "ASR/Speech": ["asr", "speech recognition", "whisper", "transcription", "voice"],
    "Translation": ["translation", "nmt", "machine translation", "mt system"],
    "Chatbot": ["chatbot", "conversational", "dialog", "chat system", "whatsapp"],
    "Benchmark": ["benchmark", "evaluation", "test set", "evaluated"],
    "Medical/Health": ["medical", "healthcare", "clinical", "health worker", "biomedical"],
    "Low-resource": ["low-resource", "under-resourced", "minority language", "underserved"],
}


def extract_methods(abstract: str) -> list[str]:
    abstract_lower = abstract.lower() if pd.notna(abstract) else ""
    methods = []
    for method, keywords in METHOD_KEYWORDS.items():
        if any(kw in abstract_lower for kw in keywords):
            methods.append(method)
    return methods


def build_domain_method_graph(df: pd.DataFrame) -> nx.Graph:
    df = df.copy()
    df["domain"] = df["domain"].replace(DOMAIN_MAP)

    domain_methods = defaultdict(set)
    for _, row in df.iterrows():
        domain = row["domain"]
        if domain not in DOMAIN_ORDER:
            continue
        methods = extract_methods(row["abstract"])
        for method in methods:
            domain_methods[domain].add(method)

    G = nx.Graph()

    for domain in DOMAIN_ORDER:
        if domain in domain_methods:
            G.add_node(domain, node_type="domain")

    all_methods = set()
    for methods in domain_methods.values():
        all_methods.update(methods)

    for method in all_methods:
        G.add_node(method, node_type="method")

    for domain, methods in domain_methods.items():
        for method in methods:
            G.add_edge(domain, method)

    return G, domain_methods


def plot_domain_method_network(G: nx.Graph, domain_methods: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))

    domain_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "domain"]
    method_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "method"]

    pos = {}
    n_domains = len(domain_nodes)
    for i, domain in enumerate(domain_nodes):
        angle = 2 * 3.14159 * i / n_domains - 3.14159 / 2
        pos[domain] = (2.5 * np.cos(angle), 2.5 * np.sin(angle))

    method_subgraph = G.subgraph(method_nodes)
    method_pos = nx.spring_layout(method_subgraph, k=1.5, iterations=50, seed=42, center=(0, 0))
    for method, (x, y) in method_pos.items():
        pos[method] = (x * 0.8, y * 0.8)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=2,
        alpha=0.35,
        edge_color=COLORS["slate_2"],
    )

    domain_colors = [DOMAIN_COLORS.get(n, COLORS["slate_3"]) for n in domain_nodes]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=domain_nodes,
        node_size=3500,
        node_color=domain_colors,
        edgecolors=COLORS["slate_4"],
        linewidths=2,
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=method_nodes,
        node_size=1200,
        node_color=COLORS["slate_1"],
        edgecolors=COLORS["slate_3"],
        linewidths=1,
        node_shape="s",
    )

    domain_labels = {n: n for n in domain_nodes}
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=domain_labels,
        font_size=16,
        font_weight="bold",
        font_family="serif",
    )

    method_labels = {n: n for n in method_nodes}
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=method_labels,
        font_size=11,
        font_family="serif",
    )

    ax.axis("off")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_method_network.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'domain_method_network.pdf'}")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers")

    G, domain_methods = build_domain_method_graph(df)

    print("\nDomain -> Methods:")
    for domain in DOMAIN_ORDER:
        if domain in domain_methods:
            methods = sorted(domain_methods[domain])
            print(f"  {domain}: {', '.join(methods)}")

    print(f"\nGraph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    plot_domain_method_network(G, domain_methods)


if __name__ == "__main__":
    main()
