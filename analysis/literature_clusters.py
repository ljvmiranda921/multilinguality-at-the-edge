"""Cluster NLP literature by abstract embeddings and extract keywords per cluster."""

from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

MAIN_DATA_PATH = ROOT / "data" / "papers_multilingual_edge_llm.csv"
APP_DATA_PATH = ROOT / "data" / "papers_application.csv"

# Color palette for clusters
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


def get_device() -> str:
    """Get the best available device (MPS for Mac, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_and_merge_data() -> pd.DataFrame:
    """Load and merge the main literature data with application papers."""
    main_df = pd.read_csv(MAIN_DATA_PATH)
    app_df = pd.read_csv(APP_DATA_PATH)

    # Normalize titles for deduplication
    def norm(s: pd.Series) -> pd.Series:
        return s.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()

    main_df["_key"] = norm(main_df["title"])
    app_df["_key"] = norm(app_df["title"])

    # Keep main_df papers, add app_df papers not already in main
    existing_keys = set(main_df["_key"])
    new_app = app_df[~app_df["_key"].isin(existing_keys)].copy()

    # For app papers without abstracts, use description
    if "abstract" not in new_app.columns:
        new_app["abstract"] = new_app.get("description", "")

    # Combine
    combined = pd.concat([main_df, new_app], ignore_index=True)
    combined = combined.drop(columns=["_key"], errors="ignore")

    # Filter to papers with abstracts
    combined = combined[combined["abstract"].notna() & (combined["abstract"].str.len() > 50)]
    combined = combined.drop_duplicates(subset=["title"])

    return combined.reset_index(drop=True)


def embed_abstracts(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed abstracts using sentence-transformers."""
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
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce dimensionality with UMAP and cluster with HDBSCAN."""
    print("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
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
    """Extract keywords for each cluster by concatenating abstracts."""
    device = get_device()
    sentence_model = SentenceTransformer(model_name, device=device)
    kw_model = KeyBERT(model=sentence_model)

    cluster_keywords = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise

        mask = labels == label
        cluster_abstracts = df.loc[mask, "abstract"].tolist()
        combined_text = " ".join(cluster_abstracts)

        # Extract keywords with diversity
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
    """Plot UMAP projection with cluster colors and keyword labels."""
    fig, ax = plt.subplots(figsize=(12, 10))

    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])

    # Plot noise points first (gray, smaller)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            coords_2d[noise_mask, 0],
            coords_2d[noise_mask, 1],
            c=COLORS["slate_2"],
            s=20,
            alpha=0.3,
            label="Unclustered",
        )

    # Plot each cluster
    for i, label in enumerate([l for l in unique_labels if l != -1]):
        mask = labels == label
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=color,
            s=50,
            alpha=0.7,
            edgecolors=COLORS["slate_4"],
            linewidths=0.3,
        )

        # Add cluster centroid label
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

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "literature_clusters_umap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'literature_clusters_umap.pdf'}")


def plot_cluster_sizes(
    labels: np.ndarray,
    cluster_keywords: dict[int, list[str]],
) -> None:
    """Plot cluster sizes as horizontal bar chart with keyword labels."""
    unique_labels = [l for l in sorted(set(labels)) if l != -1]
    sizes = [(labels == l).sum() for l in unique_labels]
    names = [", ".join(cluster_keywords.get(l, [f"Cluster {l}"])[:2]) for l in unique_labels]

    # Sort by size
    sorted_data = sorted(zip(sizes, names, unique_labels), reverse=True)
    sizes, names, unique_labels = zip(*sorted_data) if sorted_data else ([], [], [])

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.5)))
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(names))]

    y_pos = range(len(names))
    ax.barh(
        list(names)[::-1],
        list(sizes)[::-1],
        color=colors[::-1],
        edgecolor=COLORS["slate_4"],
        linewidth=0.5,
    )

    for i, (name, size) in enumerate(zip(list(names)[::-1], list(sizes)[::-1])):
        ax.text(size + 1, i, str(size), va="center", fontsize=14)

    ax.set_xlabel("Number of papers")
    ax.set_xlim(0, max(sizes) * 1.15 if sizes else 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "literature_clusters_sizes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'literature_clusters_sizes.pdf'}")


def main():
    # Load data
    df = load_and_merge_data()
    print(f"Loaded {len(df)} papers with abstracts")

    # Embed
    embeddings = embed_abstracts(df)

    # Cluster
    coords_2d, labels = cluster_embeddings(embeddings, min_cluster_size=8, min_samples=3)

    # Extract keywords
    print("Extracting keywords per cluster...")
    cluster_keywords = extract_cluster_keywords(df, labels, top_n=5)

    # Print summary
    print("\nCluster keywords:")
    for label, keywords in sorted(cluster_keywords.items()):
        count = (labels == label).sum()
        print(f"  Cluster {label} (n={count}): {', '.join(keywords)}")

    # Add cluster labels to df for inspection
    df["cluster"] = labels
    df["cluster_keywords"] = df["cluster"].map(
        lambda x: ", ".join(cluster_keywords.get(x, [])) if x != -1 else "unclustered"
    )

    # Save annotated data
    output_path = ROOT / "data" / "papers_with_clusters.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved annotated data to {output_path}")

    # Plot
    plot_clusters(coords_2d, labels, cluster_keywords, df)
    plot_cluster_sizes(labels, cluster_keywords)


if __name__ == "__main__":
    main()
