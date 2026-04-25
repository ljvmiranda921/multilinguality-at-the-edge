import argparse
import json
import random
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from adjustText import adjust_text
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS, get_device

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent
WEB_DATA_DIR = ROOT / "docs" / "assets" / "data"

plt.rcParams.update(PLOT_PARAMS)

MAIN_DATA_PATH = ROOT / "data" / "papers_multilingual_edge_llm.csv"
APP_DATA_PATH = ROOT / "data" / "papers_application.csv"

CLUSTER_COLORS = [
    COLORS["cherry"],
    COLORS["crest"],
    COLORS["purple"],
    COLORS["indigo"],
    COLORS["green"],
    COLORS["warm_blue"],
    COLORS["dark_blue"],
    COLORS["warm_purple"],
    COLORS["warm_cherry"],
    COLORS["warm_green"],
    COLORS["dark_crest"],
    COLORS["dark_purple"],
]

# fmt: off
ACCEPT_LIST = {
    "bert quantization", "smaller memory", "distillation method", 
    "thinking capabilities", "inference boost", "reasoning performance",
    "faster draft", "higher throughput", "context window",
    "prompt compression", "information entropy", "efficiency llms",
    "pruning llms", "model compression", "language tasks",
    "architecture tokenizer", "context size", "datasets diverse",
    "ai healthcare", "dialog datasets", "specialized domains",
    "encoder translation", "learn subwords", "embedding benchmark",
    # "gpu resource", "compression language", "memory challenges",
    # "corpus benchmark", "health worker", "deployed kenya", "multilingual humanitarian",
    # "budget forcing", "bigbench hard", "reasoning performance",
    # "pruning llms", "model compression",
    # "experts model", "layer router", "language combine" 
    # "embeddings semantic", "vocabulary capacity", "cultural nuances", 
    # "machine translation", "self supervised",
    # "asr", "tuning vocabulary",
    # "multilingual models", "large corpora", "continued pretraining",
    # "americas nli", "masakhaner", "bilingual llm"
    # "quantization", "bit training", "pruning", "distillation", "compression",
    # "multilingual", "low-resource", "cross-lingual",
    # "on-device", "edge", "mobile", "efficient", "lightweight",
    # "medical", "healthcare", "clinical",
    # "translation", "machine translation", "nmt",
    # "speech", "asr", "tts",
    # "bert", "transformer", "attention",
    # "fine-tuning", "adaptation", "transfer",
    # "embedding", "representation",
    # "inference", "latency", "throughput",
}

REMOVE_LIST = {
    # "language model", "language models", "large language", "models", "model",
    # "llms", "llm", "benchmark", "benchmarks", "training", "trained",
    # "datasets", "dataset", "tasks", "task", "performance", "parameters",
    # "billion parameters", "methods", "method", "approach", "approaches",
    # "results", "paper", "study", "work", "using", "based", "show",
    # "propose", "proposed", "novel", "new", "state art", "sota",
    # "experiments", "evaluation", "evaluated", "https", "github",
    # "bottleneck", "pathways", "assessed", "larger", "free structured",
    # "leveraging", "crucial", "significant", "achieve", "demonstrate",
    # "learned", "large"
}
# fmt: on


def load_and_merge_data() -> tuple[pd.DataFrame, set[str]]:
    main_df = pd.read_csv(MAIN_DATA_PATH)
    app_df = pd.read_csv(APP_DATA_PATH)

    def norm(s: pd.Series) -> pd.Series:
        return s.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()

    main_df["_key"] = norm(main_df["title"])
    app_df["_key"] = norm(app_df["title"])

    deployment_keys = set(app_df["_key"])

    existing_keys = set(main_df["_key"])
    new_app = app_df[~app_df["_key"].isin(existing_keys)].copy()

    if "abstract" not in new_app.columns:
        new_app["abstract"] = new_app.get("description", "")

    combined = pd.concat([main_df, new_app], ignore_index=True)
    combined["is_deployment"] = combined["_key"].isin(deployment_keys)
    combined = combined.drop(columns=["_key"], errors="ignore")

    combined = combined[
        combined["abstract"].notna() & (combined["abstract"].str.len() > 50)
    ]
    combined = combined.drop_duplicates(subset=["title"])

    return combined.reset_index(drop=True)


def embed_abstracts(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    device = get_device()
    print(f"Embedding abstracts using device: {device}")

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        df["abstract"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    print("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    coords_2d = reducer.fit_transform(embeddings)

    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords_2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    return coords_2d, labels


def filter_keywords(keywords: list[str], top_n: int = 5) -> list[str]:
    accepted = []

    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in REMOVE_LIST or any(term in kw_lower for term in REMOVE_LIST):
            continue
        if kw_lower in ACCEPT_LIST or any(term in kw_lower for term in ACCEPT_LIST):
            accepted.append(kw)

    if len(accepted) > top_n:
        return random.sample(accepted, top_n)
    return accepted


def extract_cluster_keywords(
    df: pd.DataFrame,
    labels: np.ndarray,
    model_name: str = "all-MiniLM-L6-v2",
    top_n: int = 5,
) -> dict[int, list[str]]:
    device = get_device()
    kw_model = KeyBERT(model=SentenceTransformer(model_name, device=device))

    cluster_keywords = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label
        cluster_abstracts = df.loc[mask, "abstract"].tolist()
        combined_text = " ".join(cluster_abstracts)

        kws = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=20,
        )
        all_kws = [kw for kw, _ in kws]
        # cluster_keywords[label] = all_kws
        cluster_keywords[label] = filter_keywords(all_kws, top_n=top_n)

    # breakpoint()

    return cluster_keywords


def plot_clusters(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    cluster_keywords: dict[int, list[str]],
    df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    unique_labels = sorted(set(labels))

    is_deployment = df["is_deployment"].values

    noise_mask = labels == -1
    if noise_mask.any():
        noise_method = noise_mask & ~is_deployment
        noise_deploy = noise_mask & is_deployment
        if noise_method.any():
            ax.scatter(
                coords_2d[noise_method, 0],
                coords_2d[noise_method, 1],
                c=COLORS["slate_2"],
                s=20,
                alpha=0.3,
                marker="o",
            )
        if noise_deploy.any():
            ax.scatter(
                coords_2d[noise_deploy, 0],
                coords_2d[noise_deploy, 1],
                c=COLORS["slate_2"],
                s=40,
                alpha=0.5,
                marker="*",
            )

    texts = []
    for i, label in enumerate([l for l in unique_labels if l != -1]):
        mask = labels == label
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        method_mask = mask & ~is_deployment
        deploy_mask = mask & is_deployment

        if method_mask.any():
            ax.scatter(
                coords_2d[method_mask, 0],
                coords_2d[method_mask, 1],
                c=color,
                s=50,
                alpha=0.7,
                marker="o",
                edgecolors=COLORS["slate_4"],
                linewidths=0.3,
            )

        if deploy_mask.any():
            ax.scatter(
                coords_2d[deploy_mask, 0],
                coords_2d[deploy_mask, 1],
                c=color,
                s=95,
                alpha=0.9,
                marker="*",
                edgecolors=COLORS["slate_4"],
                linewidths=0.5,
            )

        centroid = coords_2d[mask].mean(axis=0)
        keywords = cluster_keywords.get(label, [])
        if keywords:
            plot_kws = (
                random.sample(keywords, min(3, len(keywords)))
                if len(keywords) > 3
                else keywords[:3]
            )
            label_text = "\n".join(f"``{kw}''" for kw in plot_kws)
            t = ax.text(
                centroid[0],
                centroid[1],
                label_text,
                fontsize=16,
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            texts.append(t)

    adjust_text(texts, ax=ax)

    ax.scatter([], [], c=COLORS["slate_3"], s=50, marker="o", label="Edge ML Methods")  # fmt: skip
    ax.scatter([], [], c=COLORS["slate_3"], s=80, marker="*", label="Real-world Deployment")  # fmt: skip
    ax.legend(frameon=False, loc="lower left", fontsize=18)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "literature_clusters_umap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'literature_clusters_umap.pdf'}")


def export_web(
    df: pd.DataFrame,
    coords_2d: np.ndarray,
    labels: np.ndarray,
    cluster_keywords: dict[int, list[str]],
    outpath: Path,
    n_representatives: int = 4,
) -> None:
    """Write the cluster scatter as JSON for the website renderer.

    Representatives = the papers nearest to each cluster's centroid.
    """
    points = [
        {
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "cluster": int(labels[i]),
            "is_deployment": bool(df["is_deployment"].iloc[i]),
        }
        for i in range(len(df))
    ]

    clusters = []
    for cid, kws in sorted(cluster_keywords.items()):
        mask = labels == cid
        member_idx = np.where(mask)[0]
        cluster_coords = coords_2d[mask]
        centroid = cluster_coords.mean(axis=0)

        dists = np.linalg.norm(cluster_coords - centroid, axis=1)
        order = np.argsort(dists)[:n_representatives]
        rep_indices = member_idx[order]

        reps = []
        for idx in rep_indices:
            row = df.iloc[idx]
            url = row.get("s2_url")
            url = str(url) if pd.notna(url) else ""
            year = row.get("year")
            year = int(year) if pd.notna(year) else None
            reps.append(
                {
                    "title": str(row["title"]),
                    "url": url,
                    "year": year,
                }
            )

        clusters.append(
            {
                "id": int(cid),
                "size": int(mask.sum()),
                "keywords": kws,
                "centroid": [float(centroid[0]), float(centroid[1])],
                "representatives": reps,
            }
        )

    bounds = {
        "x_min": float(coords_2d[:, 0].min()),
        "x_max": float(coords_2d[:, 0].max()),
        "y_min": float(coords_2d[:, 1].min()),
        "y_max": float(coords_2d[:, 1].max()),
    }

    payload = {"points": points, "clusters": clusters, "bounds": bounds}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(points)} points + {len(clusters)} clusters to {outpath}")


def main(export_to_web: bool = False):
    # Load data
    df = load_and_merge_data()
    print(f"Loaded {len(df)} papers with abstracts")
    model_name = "all-MiniLM-L6-v2"
    random_state = 42
    embeddings = embed_abstracts(df, model_name=model_name)
    coords_2d, labels = cluster_embeddings(
        embeddings,
        min_cluster_size=8,
        min_samples=3,
        random_state=random_state,
    )

    cluster_keywords = extract_cluster_keywords(df, labels, top_n=5, model_name=model_name)  # fmt: skip
    print("\nCluster keywords:")
    for label, keywords in sorted(cluster_keywords.items()):
        count = (labels == label).sum()
        kw_str = ", ".join(f'"{kw}"' for kw in keywords)
        print(f"  Cluster {label} (n={count}): {kw_str}")

    df["cluster"] = labels
    df["cluster_keywords"] = df["cluster"].map(lambda x: ", ".join(cluster_keywords.get(x, [])) if x != -1 else "unclustered")  # fmt: skip
    df["umap_x"] = coords_2d[:, 0]
    df["umap_y"] = coords_2d[:, 1]

    output_path = ROOT / "data" / "papers_with_clusters.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved annotated data to {output_path}")

    print("\nCluster sizes:")
    for label, keywords in sorted(cluster_keywords.items()):
        count = (labels == label).sum()
        kw_str = ", ".join(f'"{kw}"' for kw in keywords[:2])
        print(f"  {kw_str}: {count}")

    plot_clusters(coords_2d, labels, cluster_keywords, df)

    if export_to_web:
        export_web(
            df,
            coords_2d,
            labels,
            cluster_keywords,
            WEB_DATA_DIR / "literature_clusters.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export_to_web",
        action="store_true",
        help="Also export cluster + point data as JSON.",
    )
    args = parser.parse_args()
    main(export_to_web=args.export_to_web)
