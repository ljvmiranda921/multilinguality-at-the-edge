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
            top_n=top_n,
        )
        cluster_keywords[label] = [kw for kw, _ in kws]

    return cluster_keywords


def plot_clusters(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    cluster_keywords: dict[int, list[str]],
    df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

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
                marker="x",
            )

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
                s=80,
                alpha=0.9,
                marker="x",
                edgecolors=COLORS["slate_4"],
                linewidths=0.5,
            )

        centroid = coords_2d[mask].mean(axis=0)
        keywords = cluster_keywords.get(label, [])
        if keywords:
            label_text = ", ".join(keywords[:2])
            ax.annotate(
                label_text,
                centroid,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax.scatter([], [], c=COLORS["slate_3"], s=50, marker="o", label="Method")
    ax.scatter([], [], c=COLORS["slate_3"], s=80, marker="X", label="Deployment")
    ax.legend(frameon=False, loc="upper right", fontsize=14)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "literature_clusters_umap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'literature_clusters_umap.pdf'}")


def main():
    # Load data
    df = load_and_merge_data()
    print(f"Loaded {len(df)} papers with abstracts")
    model_name = "all-MiniLM-L12-v2"
    random_state = 21
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
        print(f"  Cluster {label} (n={count}): {', '.join(keywords)}")

    df["cluster"] = labels
    df["cluster_keywords"] = df["cluster"].map(lambda x: ", ".join(cluster_keywords.get(x, [])) if x != -1 else "unclustered")  # fmt: skip

    output_path = ROOT / "data" / "papers_with_clusters.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved annotated data to {output_path}")

    print("\nCluster sizes:")
    for label, keywords in sorted(cluster_keywords.items()):
        count = (labels == label).sum()
        print(f"  {', '.join(keywords[:2])}: {count}")

    plot_clusters(coords_2d, labels, cluster_keywords, df)


if __name__ == "__main__":
    main()
