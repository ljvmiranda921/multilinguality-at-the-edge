from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from analysis.utils import COLORS, OUTPUT_DIR, PLOT_PARAMS, get_device

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

DATA_PATH = ROOT / "data" / "papers_application.csv"

STOPWORDS = {
    "language", "languages", "model", "models", "llm", "llms", "large",
    "multilingual", "paper", "study", "approach", "method", "methods",
    "using", "based", "performance", "results", "data", "task", "tasks",
}


def extract_keywords_per_paper(df: pd.DataFrame, top_n: int = 5) -> dict[int, list[str]]:
    device = get_device()
    kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2", device=device))

    paper_keywords = {}
    for idx, row in df.iterrows():
        abstract = row["abstract"] if pd.notna(row["abstract"]) else ""
        if len(abstract) < 50:
            continue

        kws = kw_model.extract_keywords(
            abstract,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=top_n * 2,
        )

        filtered = []
        for kw, score in kws:
            kw_lower = kw.lower()
            if not any(sw in kw_lower for sw in STOPWORDS):
                filtered.append(kw_lower)
            if len(filtered) >= top_n:
                break

        paper_keywords[idx] = filtered

    return paper_keywords


def build_cooccurrence(paper_keywords: dict[int, list[str]]) -> Counter:
    cooccur = Counter()
    for keywords in paper_keywords.values():
        for kw1, kw2 in combinations(sorted(set(keywords)), 2):
            cooccur[(kw1, kw2)] += 1
    return cooccur


def plot_keyword_network(cooccur: Counter, min_count: int = 1) -> None:
    G = nx.Graph()

    for (kw1, kw2), count in cooccur.items():
        if count >= min_count:
            G.add_edge(kw1, kw2, weight=count)

    if len(G.nodes()) == 0:
        print("No edges meet the min_count threshold")
        return

    degrees = dict(G.degree())
    node_sizes = [300 + degrees[n] * 150 for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    edges = G.edges(data=True)
    weights = [e[2]["weight"] for e in edges]
    max_weight = max(weights) if weights else 1

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[1 + (w / max_weight) * 3 for w in weights],
        alpha=0.4,
        edge_color=COLORS["slate_2"],
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=COLORS["warm_blue"],
        edgecolors=COLORS["slate_4"],
        linewidths=1,
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=10,
        font_family="serif",
    )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "keyword_cooccurrence_network.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR / 'keyword_cooccurrence_network.pdf'}")


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers")

    print("Extracting keywords...")
    paper_keywords = extract_keywords_per_paper(df, top_n=5)

    print("\nKeywords per paper:")
    for idx, kws in list(paper_keywords.items())[:5]:
        print(f"  {df.loc[idx, 'title'][:50]}...: {kws}")

    cooccur = build_cooccurrence(paper_keywords)
    print(f"\nTop co-occurrences:")
    for (kw1, kw2), count in cooccur.most_common(15):
        print(f"  {kw1} + {kw2}: {count}")

    plot_keyword_network(cooccur, min_count=1)


if __name__ == "__main__":
    main()
