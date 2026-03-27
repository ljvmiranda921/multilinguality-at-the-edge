from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from adjustText import adjust_text

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

# Keywords are organized by technique/approach type to avoid mixing methods
# with application topics. Each keyword list uses specific multi-word phrases
# to reduce false positives (e.g., "retrieval-augmented" instead of "retrieval").
TECHNIQUE_KEYWORDS = {
    # Efficiency & compression techniques
    "RAG": ["retrieval-augmented", "retrieval augmented", " rag "],
    "LoRA/QLoRA": ["lora", "qlora", "peft"],
    "Quantization": ["quantiz", "int8", "int4", "ptq", "gguf"],
    "Distillation": ["distil", "student model", "teacher model"],
    "Fine-tuning": ["fine-tun", "finetuning", "instruction tuning", " sft "],
    "MoE": ["mixture of experts", " moe ", "language family expert"],
    "Federated Learning": ["federated learning", "federated few-shot"],
    "Synthetic Data": ["synthetic data", "data augmentation", "synthetic qa"],
    "Continual Pretraining": [
        "continual pretrain",
        "continued pretrain",
        "further pretrain",
        "pre-trained on",
    ],
    # Deployment modalities
    "On-device/Edge": [
        "on-device",
        "edge deploy",
        "on device",
        "edge asr",
        "neural engine",
    ],
    "ASR": ["asr", "speech recognition", "whisper", "transcription"],
    "Machine Translation": ["machine translation", " nmt ", " mt system", " mt model"],
    "Chatbot": ["chatbot", "conversational", "chat system", "whatsapp"],
    # Cross-cutting concerns
    "Benchmark": ["benchmark", "test set", "evaluation benchmark"],
    "Low-resource NLP": [
        "low-resource",
        "under-resourced",
        "minority language",
        "underserved",
    ],
}


def extract_techniques(text: str) -> list[str]:
    """Extract techniques from abstract + description combined text."""
    text_lower = text.lower() if pd.notna(text) else ""
    techniques = []
    for technique, keywords in TECHNIQUE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            techniques.append(technique)
    return techniques


def build_domain_technique_graph(
    df: pd.DataFrame,
) -> tuple[nx.Graph, dict, dict]:
    df = df.copy()
    df["domain"] = df["domain"].replace(DOMAIN_MAP)

    # Combine abstract and description for richer matching
    df["combined_text"] = df["abstract"].fillna("") + " " + df["description"].fillna("")

    # Track both set membership and paper counts for edge weights
    domain_techniques: dict[str, set] = defaultdict(set)
    edge_weights: dict[tuple[str, str], int] = defaultdict(int)

    for _, row in df.iterrows():
        domain = row["domain"]
        if domain not in DOMAIN_ORDER:
            continue
        techniques = extract_techniques(row["combined_text"])
        for technique in techniques:
            domain_techniques[domain].add(technique)
            edge_weights[(domain, technique)] += 1

    G = nx.Graph()

    for domain in DOMAIN_ORDER:
        if domain in domain_techniques:
            G.add_node(domain, node_type="domain")

    all_techniques = set()
    for techniques in domain_techniques.values():
        all_techniques.update(techniques)

    for technique in all_techniques:
        # Count how many domains connect to this technique
        degree = sum(1 for d in domain_techniques if technique in domain_techniques[d])
        G.add_node(technique, node_type="technique", degree=degree)

    for (domain, technique), weight in edge_weights.items():
        G.add_edge(domain, technique, weight=weight)

    return G, domain_techniques, edge_weights


def plot_domain_technique_network(
    G: nx.Graph,
    domain_techniques: dict,
    edge_weights: dict,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    domain_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "domain"]
    technique_nodes = [
        n for n, d in G.nodes(data=True) if d.get("node_type") == "technique"
    ]

    # Position domains on an outer ring
    pos = {}
    n_domains = len(domain_nodes)
    radius = 4.5
    for i, domain in enumerate(domain_nodes):
        angle = 2 * np.pi * i / n_domains - np.pi / 2
        pos[domain] = (radius * np.cos(angle), radius * np.sin(angle))

    # Initialize technique positions as weighted centroid of connected domains
    # Weight by edge weight so techniques are pulled toward their primary domain
    init_pos = {}
    for technique in technique_nodes:
        neighbors = list(G.neighbors(technique))
        if neighbors:
            weights = [
                edge_weights.get((n, technique), edge_weights.get((technique, n), 1))
                for n in neighbors
                if n in pos
            ]
            xs = [pos[n][0] for n in neighbors if n in pos]
            ys = [pos[n][1] for n in neighbors if n in pos]
            if xs:
                total_w = sum(weights)
                wx = sum(x * w for x, w in zip(xs, weights)) / total_w
                wy = sum(y * w for y, w in zip(ys, weights)) / total_w
                # Pull toward center but not all the way (0.5 factor)
                init_pos[technique] = (wx * 0.5, wy * 0.5)
            else:
                init_pos[technique] = (0, 0)
        else:
            init_pos[technique] = (0, 0)

    # Spring layout with fixed domain positions
    fixed_pos = {**pos, **init_pos}
    full_pos = nx.spring_layout(
        G,
        pos=fixed_pos,
        fixed=domain_nodes,
        k=1.8,
        iterations=200,
        seed=42,
    )

    # Clamp technique nodes: high-degree nodes pulled closer to center
    max_degree = max(G.nodes[t].get("degree", 1) for t in technique_nodes)
    for t in technique_nodes:
        x, y = full_pos[t]
        degree = G.nodes[t].get("degree", 1)
        # High degree → tighter max distance (closer to center)
        frac = degree / max_degree
        max_dist = radius * (0.75 - 0.35 * frac)
        dist = np.sqrt(x**2 + y**2)
        if dist > max_dist:
            scale = max_dist / dist
            full_pos[t] = (x * scale, y * scale)

    pos.update({t: full_pos[t] for t in technique_nodes})

    # Draw edges colored by domain, with width proportional to weight
    for (u, v), weight in edge_weights.items():
        domain_node = u if u in domain_nodes else v
        color = DOMAIN_COLORS.get(domain_node, COLORS["slate_3"])
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=[(u, v)],
            width=1.0 + weight * 1.5,
            alpha=0.4,
            edge_color=[color],
        )

    # Draw domain nodes: large colored circles, sized by total paper count
    domain_colors = [DOMAIN_COLORS.get(n, COLORS["slate_3"]) for n in domain_nodes]
    domain_sizes = []
    for d in domain_nodes:
        total_papers = sum(w for (dom, _), w in edge_weights.items() if dom == d)
        domain_sizes.append(4000 + total_papers * 500)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=domain_nodes,
        node_size=domain_sizes,
        node_color=domain_colors,
        edgecolors="white",
        linewidths=2.5,
        alpha=0.55,
    )

    # Draw technique nodes: higher degree = larger, darker, and more central
    max_degree = max(G.nodes[t].get("degree", 1) for t in technique_nodes)
    technique_sizes = []
    technique_colors = []
    for t in technique_nodes:
        degree = G.nodes[t].get("degree", 1)
        technique_sizes.append(300 + degree * 250)
        # Interpolate gray: high degree = dark (slate_4), low = light (slate_2)
        frac = degree / max_degree
        r_lo, g_lo, b_lo = 0.71, 0.74, 0.78  # slate_2 approx
        r_hi, g_hi, b_hi = 0.14, 0.16, 0.19  # slate_4 approx
        r = r_lo + (r_hi - r_lo) * frac
        g = g_lo + (g_hi - g_lo) * frac
        b = b_lo + (b_hi - b_lo) * frac
        technique_colors.append((r, g, b))

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=technique_nodes,
        node_size=technique_sizes,
        node_color=technique_colors,
        edgecolors="none",
        alpha=0.85,
    )

    # Domain labels: bold, white text on the colored node
    for domain in domain_nodes:
        x, y = pos[domain]
        ax.text(
            x,
            y,
            domain,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            fontfamily="serif",
            color=COLORS["slate_4"],
            zorder=5,
        )

    # Technique labels: placed with adjustText to avoid overlaps
    technique_texts = []
    for technique in technique_nodes:
        x, y = pos[technique]
        t = ax.text(
            x,
            y,
            technique,
            ha="center",
            va="center",
            fontsize=18,
            fontfamily="serif",
            color=COLORS["slate_4"],
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=COLORS["slate_2"],
                linewidth=0.5,
                alpha=0.9,
            ),
            zorder=4,
        )
        technique_texts.append(t)

    adjust_text(
        technique_texts,
        ax=ax,
        expand=(1.5, 1.5),
        force_text=(0.5, 0.5),
        force_points=(0.3, 0.3),
    )

    ax.axis("off")
    margin = 1.0
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_method_network.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "domain_method_network.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'domain_method_network.pdf'}")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers")

    G, domain_techniques, edge_weights = build_domain_technique_graph(df)

    print("\nDomain -> Techniques:")
    for domain in DOMAIN_ORDER:
        if domain in domain_techniques:
            techniques = sorted(domain_techniques[domain])
            print(f"  {domain}: {', '.join(techniques)}")

    print(f"\nGraph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print("\nEdge weights:")
    for (d, t), w in sorted(edge_weights.items()):
        print(f"  {d} -- {t}: {w}")

    plot_domain_technique_network(G, domain_techniques, edge_weights)


if __name__ == "__main__":
    main()
