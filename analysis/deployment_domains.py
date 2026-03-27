from pathlib import Path

import pandas as pd

from analysis.utils import OUTPUT_DIR

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

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


METHOD_KEYWORDS = {
    "RAG": ["rag", "retrieval-augmented", "retrieval augmented"],
    "LoRA": ["lora", "qlora"],
    "Quantization": ["quantiz", "int8", "int4", "ptq"],
    "Distillation": ["distil"],
    "Fine-tuning": ["fine-tun", "finetuning"],
    "MoE": ["mixture of experts", "moe", "language family expert"],
    "Federated": ["federated"],
    "Synthetic data": ["synthetic data", "data augmentation"],
    "Continual pretraining": ["continual pretrain", "continued pretrain", "further pretrain"],
}


def extract_method_tags(abstract: str) -> list[str]:
    abstract_lower = abstract.lower() if pd.notna(abstract) else ""
    tags = []
    for method, keywords in METHOD_KEYWORDS.items():
        if any(kw in abstract_lower for kw in keywords):
            tags.append(method)
    return tags


def generate_markdown_table(df: pd.DataFrame) -> str:
    df = df.copy()
    df["domain"] = df["domain"].replace(DOMAIN_MAP)

    lines = []
    lines.append("| Domain | Example Applications |")
    lines.append("|--------|---------------------|")

    for domain in DOMAIN_ORDER:
        domain_df = df[df["domain"] == domain]
        if domain_df.empty:
            continue

        examples = []
        for _, row in domain_df.iterrows():
            methods = extract_method_tags(row["abstract"])
            method_str = "/".join(methods) if methods else ""
            
            desc = row["description"]
            if pd.isna(desc) or len(desc) < 10:
                desc = row["title"]

            if method_str:
                short_desc = f"{method_str}-based {desc.lower()}"
            else:
                short_desc = desc

            url = row["url"]
            examples.append(f"{short_desc} ({url})")

        cell = "; ".join(examples)
        lines.append(f"| {domain} | {cell} |")

    return "\n".join(lines)


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} papers across {df['domain'].nunique()} domains")

    md = generate_markdown_table(df)
    print("\n" + md)

    output_path = OUTPUT_DIR / "deployment_domains_table.md"
    with open(output_path, "w") as f:
        f.write(md)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
