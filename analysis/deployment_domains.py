from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.path as mpath
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
    "Climate": COLORS["warm_blue"],
    "Finance": COLORS["green"],
    "Healthcare": COLORS["warm_cherry"],
    "Legal": COLORS["slate_2"],
    "Social": COLORS["crest"],
    "Speech": COLORS["warm_purple"],
}

TECHNIQUE_KEYWORDS = {
    # "RAG": ["retrieval-augmented", "retrieval augmented", " rag "],
    "Data\nCuration": [
        "data curation",
        "data collectors",
        "dataset of",
        "multilingual dataset",
        "multilingual corpus",
        "multilingual collection",
        "court rulings with summaries",
    ],
    "LoRA/\nQLoRA": ["lora", "qlora", "peft"],
    "Quantization": ["quantiz", "int8", "int4", "ptq", "gguf"],
    "Distillation": ["distil", "student model", "teacher model"],
    "Supervised\nFine-tuning": [
        "fine-tun",
        "finetuning",
        "instruction tuning",
        " sft ",
    ],
    "Mixtures-of-Experts": ["mixture of experts", " moe ", "language family expert"],
    "Federated\nLearning": ["federated learning", "federated few-shot"],
    "Synthetic\nData": ["synthetic data", "data augmentation", "synthetic qa"],
    "Continual\nPretraining": [
        "continual pretrain",
        "continued pretrain",
        "further pretrain",
        "pre-trained on",
    ],
    "ASR": ["asr", "speech recognition", "whisper", "transcription"],
    "Machine\nTranslation": ["machine translation", " nmt ", " mt system", " mt model"],
    "Dialogue\nSystems": ["chatbot", "conversational", "chat system", "whatsapp"],
    "Benchmarking": ["benchmark", "test set", "evaluation benchmark"],
    # "Low-resource\nNLP": [
    #     "low-resource",
    #     "under-resourced",
    #     "minority language",
    #     "underserved",
    # ],
}


def extract_techniques(text: str) -> list[str]:
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
    df["combined_text"] = df["abstract"].fillna("") + " " + df["description"].fillna("")

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
    fig, ax = plt.subplots(figsize=(10, 12))

    domain_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "domain"]
    technique_nodes = [
        n for n, d in G.nodes(data=True) if d.get("node_type") == "technique"
    ]

    pos = {}
    n_domains = len(domain_nodes)
    radius = 4.5
    for i, domain in enumerate(domain_nodes):
        angle = 2 * np.pi * i / n_domains - np.pi / 2
        pos[domain] = (radius * np.cos(angle), radius * np.sin(angle))

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
                degree = G.nodes[technique].get("degree", 1)
                pull = 0.20 - 0.15 * (
                    degree / max(G.nodes[t].get("degree", 1) for t in technique_nodes)
                )
                init_pos[technique] = (wx * pull, wy * pull)
            else:
                init_pos[technique] = (0, 0)
        else:
            init_pos[technique] = (0, 0)

    fixed_pos = {**pos, **init_pos}
    full_pos = nx.spring_layout(
        G,
        pos=fixed_pos,
        fixed=domain_nodes,
        k=2.5,
        iterations=300,
        seed=42,
    )

    # Re-center technique cloud
    tech_xs = [full_pos[t][0] for t in technique_nodes]
    tech_ys = [full_pos[t][1] for t in technique_nodes]
    cx_off = np.mean(tech_xs)
    cy_off = np.mean(tech_ys)
    for t in technique_nodes:
        x, y = full_pos[t]
        full_pos[t] = (x - cx_off, y - cy_off)

    # Push apart overlapping technique nodes
    min_separation = 0.6
    for _ in range(50):
        moved = False
        for i, t1 in enumerate(technique_nodes):
            for t2 in technique_nodes[i + 1 :]:
                x1, y1 = full_pos[t1]
                x2, y2 = full_pos[t2]
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                if dist < min_separation and dist > 0:
                    push = (min_separation - dist) / 2
                    nx_dir = dx / dist
                    ny_dir = dy / dist
                    full_pos[t1] = (x1 - nx_dir * push, y1 - ny_dir * push)
                    full_pos[t2] = (x2 + nx_dir * push, y2 + ny_dir * push)
                    moved = True
        if not moved:
            break

    # Clamp: high-degree nodes closer to center
    max_degree = max(G.nodes[t].get("degree", 1) for t in technique_nodes)
    for t in technique_nodes:
        x, y = full_pos[t]
        degree = G.nodes[t].get("degree", 1)
        frac = degree / max_degree
        max_dist = radius * (0.65 - 0.30 * frac)
        dist = np.sqrt(x**2 + y**2)
        if dist > max_dist:
            scale = max_dist / dist
            full_pos[t] = (x * scale, y * scale)

    pos.update({t: full_pos[t] for t in technique_nodes})

    # Bezier curve edges colored by domain
    for (u, v), weight in edge_weights.items():
        domain_node = u if u in domain_nodes else v
        color = DOMAIN_COLORS.get(domain_node, COLORS["slate_3"])
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx_perp, ny_perp = -dy / length, dx / length
        else:
            nx_perp, ny_perp = 0, 0
        curvature = 0.2 * length
        sign = np.sign(mx * ny_perp - my * nx_perp)
        cx = (mx + nx_perp * curvature * sign) * 0.85
        cy = (my + ny_perp * curvature * sign) * 0.85
        path = mpath.Path(
            [(x0, y0), (cx, cy), (x1, y1)],
            [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3],
        )
        patch = mpatches.FancyArrowPatch(
            path=path,
            arrowstyle="-",
            linewidth=1.0 + weight * 1.2,
            color=color,
            alpha=0.35,
            zorder=1,
        )
        ax.add_patch(patch)

    # Domain nodes
    domain_colors = [DOMAIN_COLORS.get(n, COLORS["slate_3"]) for n in domain_nodes]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=domain_nodes,
        node_size=5000,
        node_color=domain_colors,
        edgecolors="white",
        linewidths=2.5,
        alpha=0.85,
    )

    # Technique nodes: degree -> size and darkness
    technique_sizes = []
    technique_colors = []
    for t in technique_nodes:
        degree = G.nodes[t].get("degree", 1)
        technique_sizes.append(300 + degree * 250)
        frac = degree / max_degree
        r_lo, g_lo, b_lo = 0.71, 0.74, 0.78
        r_hi, g_hi, b_hi = 0.14, 0.16, 0.19
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

    # Domain labels
    domain_texts = []
    for domain in domain_nodes:
        x, y = pos[domain]
        dt = ax.text(
            x,
            y,
            r"\textsc{" + domain + r"}",
            ha="center",
            va="center",
            fontsize=28,
            fontfamily="serif",
            color="black",
            zorder=5,
        )
        domain_texts.append(dt)

    # Technique labels with arrows
    technique_texts = []
    for technique in technique_nodes:
        x, y = pos[technique]
        t = ax.text(
            x,
            y - 0.1,
            technique,
            ha="center",
            va="center",
            ma="left",
            fontsize=24,
            fontfamily="serif",
            color="black",
            zorder=4,
        )
        technique_texts.append(t)

    adjust_text(
        technique_texts,
        ax=ax,
        add_objects=domain_texts,
        expand=(1.2, 1.2),
        force_text=(0.3, 0.3),
        force_points=(0.65, 0.75),
        arrowprops=dict(arrowstyle="-", color=COLORS["slate_2"], linewidth=0.8),
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
