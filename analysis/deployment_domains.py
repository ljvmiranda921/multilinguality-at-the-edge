import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text

from analysis.utils import (
    COLORS,
    OUTPUT_DIR,
    PLOT_PARAMS,
    WEB_COLORS,
    WEB_FIGURES_DIR,
    WEB_PLOT_PARAMS,
)

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_application.csv"
WEB_DATA_DIR = ROOT / "docs" / "assets" / "data"

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

WEB_DOMAIN_COLORS = {
    "Agriculture": WEB_COLORS["warm"],
    "Climate": WEB_COLORS["cool"],
    "Finance": WEB_COLORS["accent"],
    "Healthcare": WEB_COLORS["warm_light"],
    "Legal": WEB_COLORS["muted"],
    "Social": WEB_COLORS["accent"],
    "Speech": WEB_COLORS["cool"],
}

WEB_DOMAIN_LABELS = {
    "Agriculture": "Agri\nculture",
    "Healthcare": "Health\ncare",
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


def _compute_network_layout(
    G: nx.Graph, edge_weights: dict[tuple[str, str], int]
) -> tuple[list[str], list[str], dict[str, tuple[float, float]], int]:
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

    tech_xs = [full_pos[t][0] for t in technique_nodes]
    tech_ys = [full_pos[t][1] for t in technique_nodes]
    cx_off = np.mean(tech_xs)
    cy_off = np.mean(tech_ys)
    for t in technique_nodes:
        x, y = full_pos[t]
        full_pos[t] = (x - cx_off, y - cy_off)

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

    min_separation = 1.2
    for _ in range(200):
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

    pos.update({t: full_pos[t] for t in technique_nodes})
    return domain_nodes, technique_nodes, pos, max_degree


def _build_domain_samples(
    df: pd.DataFrame, max_per_domain: int = 2
) -> dict[str, list[dict]]:
    records = df.copy()
    records["domain"] = records["domain"].replace(DOMAIN_MAP)
    records["year"] = pd.to_numeric(records.get("year"), errors="coerce")
    records = records.sort_values("year", ascending=False, na_position="last")

    out: dict[str, list[dict]] = {}
    for domain in DOMAIN_ORDER:
        subset = records[records["domain"] == domain].head(max_per_domain)
        samples = []
        for _, row in subset.iterrows():
            url = row.get("url")
            url = str(url) if pd.notna(url) else ""
            year = row.get("year")
            year = int(year) if pd.notna(year) else None
            samples.append(
                {
                    "title": str(row.get("title", "")),
                    "url": url,
                    "year": year,
                }
            )
        out[domain] = samples
    return out


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

    # adjust the cloud
    tech_xs = [full_pos[t][0] for t in technique_nodes]
    tech_ys = [full_pos[t][1] for t in technique_nodes]
    cx_off = np.mean(tech_xs)
    cy_off = np.mean(tech_ys)
    for t in technique_nodes:
        x, y = full_pos[t]
        full_pos[t] = (x - cx_off, y - cy_off)

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

    min_separation = 1.2
    for _ in range(200):
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

    pos.update({t: full_pos[t] for t in technique_nodes})

    # better curves: basically draw a simple bezier curve
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

    technique_sizes = []
    technique_colors = []
    for t in technique_nodes:
        degree = G.nodes[t].get("degree", 1)
        technique_sizes.append(300 + degree * 250)
        frac = degree / max_degree
        # use cam colors
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


def plot_domain_technique_network_web(
    G: nx.Graph,
    domain_techniques: dict,
    edge_weights: dict,
    outpath: Path,
) -> None:
    del domain_techniques
    with plt.rc_context(WEB_PLOT_PARAMS):
        fig, ax = plt.subplots(figsize=(7.2, 7.2))

        domain_nodes, technique_nodes, pos, max_degree = _compute_network_layout(
            G, edge_weights
        )

        for (u, v), weight in edge_weights.items():
            domain_node = u if u in domain_nodes else v
            color = WEB_DOMAIN_COLORS.get(domain_node, WEB_COLORS["cool"])
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
                linewidth=0.8 + weight * 0.9,
                color=color,
                alpha=0.26,
                zorder=1,
            )
            ax.add_patch(patch)

        domain_colors = [
            WEB_DOMAIN_COLORS.get(n, WEB_COLORS["cool"]) for n in domain_nodes
        ]
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=domain_nodes,
            node_size=4300,
            node_color=domain_colors,
            edgecolors=WEB_COLORS["ink"],
            linewidths=1.2,
            alpha=0.95,
        )

        technique_sizes = []
        technique_colors = {}
        for t in technique_nodes:
            degree = G.nodes[t].get("degree", 1)
            technique_sizes.append(320 + degree * 210)
            frac = degree / max_degree
            r_lo, g_lo, b_lo = 0.84, 0.87, 0.91
            r_hi, g_hi, b_hi = 0.21, 0.19, 0.18
            r = r_lo + (r_hi - r_lo) * frac
            g = g_lo + (g_hi - g_lo) * frac
            b = b_lo + (b_hi - b_lo) * frac
            technique_colors[t] = (r, g, b)

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=technique_nodes,
            node_size=technique_sizes,
            node_color=[technique_colors[t] for t in technique_nodes],
            edgecolors=WEB_COLORS["white"],
            linewidths=0.8,
            alpha=0.95,
        )

        for domain in domain_nodes:
            x, y = pos[domain]
            label = WEB_DOMAIN_LABELS.get(domain, domain)
            label_fontsize = 11 if "\n" in label else 12
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                fontfamily="Tomato Grotesk",
                fontweight="bold",
                color=WEB_COLORS["white"],
                linespacing=0.95,
                zorder=5,
            )

        for i, technique in enumerate(technique_nodes):
            x, y = pos[technique]
            dist = np.hypot(x, y)
            if dist < 0.25:
                angle = 2 * np.pi * i / max(len(technique_nodes), 1)
                ux, uy = np.cos(angle), np.sin(angle)
            else:
                ux, uy = x / dist, y / dist
            tx = x + ux * 0.62
            ty = y + uy * 0.62
            if ux > 0.2:
                ha = "left"
            elif ux < -0.2:
                ha = "right"
            else:
                ha = "center"
            ax.annotate(
                technique,
                xy=(x, y),
                xytext=(tx, ty),
                textcoords="data",
                ha=ha,
                va="center",
                fontsize=9.3,
                fontfamily="Univers",
                color=WEB_COLORS["ink"],
                arrowprops={
                    "arrowstyle": "-",
                    "color": WEB_COLORS["muted"],
                    "linewidth": 0.75,
                    "alpha": 0.45,
                    "shrinkA": 4,
                    "shrinkB": 4,
                },
                zorder=4,
            )

        ax.axis("off")
        margin = 1.8
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"Saved to {outpath}")


def export_domain_network_web_data(
    G: nx.Graph,
    edge_weights: dict[tuple[str, str], int],
    df: pd.DataFrame,
    outpath: Path,
) -> None:
    domain_nodes, technique_nodes, pos, max_degree = _compute_network_layout(
        G, edge_weights
    )
    samples = _build_domain_samples(df, max_per_domain=2)

    domains = []
    for domain in domain_nodes:
        x, y = pos[domain]
        domains.append(
            {
                "id": domain,
                "label": WEB_DOMAIN_LABELS.get(domain, domain),
                "x": float(x),
                "y": float(y),
                "color": WEB_DOMAIN_COLORS.get(domain, WEB_COLORS["cool"]),
                "samples": samples.get(domain, []),
            }
        )

    techniques = []
    for i, technique in enumerate(technique_nodes):
        x, y = pos[technique]
        degree = int(G.nodes[technique].get("degree", 1))
        size = 320 + degree * 210

        dist = np.hypot(x, y)
        if dist < 0.25:
            angle = 2 * np.pi * i / max(len(technique_nodes), 1)
            ux, uy = np.cos(angle), np.sin(angle)
        else:
            ux, uy = x / dist, y / dist

        tx = x + ux * 0.68
        ty = y + uy * 0.68
        if ux > 0.2:
            anchor = "start"
        elif ux < -0.2:
            anchor = "end"
        else:
            anchor = "middle"

        frac = degree / max_degree
        r_lo, g_lo, b_lo = 0.84, 0.87, 0.91
        r_hi, g_hi, b_hi = 0.21, 0.19, 0.18
        r = r_lo + (r_hi - r_lo) * frac
        g = g_lo + (g_hi - g_lo) * frac
        b = b_lo + (b_hi - b_lo) * frac
        color = "#{:02x}{:02x}{:02x}".format(
            int(np.clip(r, 0, 1) * 255),
            int(np.clip(g, 0, 1) * 255),
            int(np.clip(b, 0, 1) * 255),
        )

        techniques.append(
            {
                "id": technique,
                "x": float(x),
                "y": float(y),
                "degree": degree,
                "size": int(size),
                "label_x": float(tx),
                "label_y": float(ty),
                "label_anchor": anchor,
                "color": color,
            }
        )

    edges = []
    for (u, v), weight in edge_weights.items():
        domain = u if u in domain_nodes else v
        technique = v if domain == u else u
        edges.append(
            {
                "domain": domain,
                "technique": technique,
                "weight": int(weight),
                "color": WEB_DOMAIN_COLORS.get(domain, WEB_COLORS["cool"]),
            }
        )

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    payload = {
        "domains": domains,
        "techniques": techniques,
        "edges": edges,
        "bounds": {
            "x_min": float(min(all_x)),
            "x_max": float(max(all_x)),
            "y_min": float(min(all_y)),
            "y_max": float(max(all_y)),
        },
    }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {outpath}")


def main(export_to_web: bool = False):
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
    if export_to_web:
        plot_domain_technique_network_web(
            G,
            domain_techniques,
            edge_weights,
            WEB_FIGURES_DIR / "domain_method_network.svg",
        )
        export_domain_network_web_data(
            G,
            edge_weights,
            df,
            WEB_DATA_DIR / "domain_method_network.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export_to_web",
        action="store_true",
        help="Also export an SVG to docs/assets/figures/.",
    )
    args = parser.parse_args()
    main(export_to_web=args.export_to_web)
