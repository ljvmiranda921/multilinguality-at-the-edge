from typing import List, Literal

from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are an expert research paper annotator specializing in NLP and machine learning.
Given the title and abstract of a research paper, classify it according to several dimensions
relevant to multilingual and efficient NLP for edge deployment.
Your responses must strictly adhere to the specified response schema without adding any additional commentary."""

USER_PROMPT = """Title: {title}

Abstract: {abstract}

Please classify this paper according to the following dimensions:
1. Pipeline stage(s) the paper addresses (Data Collection, Pretraining, Post-training, Inference, Evaluation).
2. Primary topic(s) or techniques used in the paper.
3. Primary subject area of the paper based on ACL 2025 Subject Areas.
4. Modality of the work (e.g., text, speech, multimodal).
5. Languages studied or supported (if applicable). Use ISO 639-1 codes where possible (e.g., en, fr, de, es). Use "multilingual" if >10 languages.
6. Names of any models released by the authors (empty list if none).
7. Parameter sizes of the models released in billions (empty list if none or not specified).
8. Is this paper relevant in the context of multilingual NLP for edge devices? Score from 1 to 5.
9. State your reason for the relevance score.
"""


class ResearchPaperAnnotation(BaseModel):
    pipeline_stages: List[
        Literal[
            "Data Collection",
            "Pretraining",
            "Post-training",
            "Inference",
            "Evaluation",
        ]
    ] = Field(
        ...,
        description=(
            "Which stage(s) of the ML pipeline does this paper address? "
            "Data Collection: corpus creation, data augmentation, annotation. "
            "Pretraining: training from scratch, continued pretraining, foundation models. "
            "Post-training: fine-tuning, instruction tuning, alignment, adapters, LoRA. "
            "Inference: quantization, pruning, distillation, deployment, on-device. "
            "Evaluation: benchmarks, metrics, leaderboards, analysis."
        ),
    )
    topics: List[
        Literal[
            "Quantization",
            "Pruning",
            "Knowledge Distillation",
            "Model Compression",
            "Neural Architecture Search",
            "Low-Rank Factorization",
            "Parameter-Efficient Fine-Tuning",
            "Efficient Architectures",
            "Data-Efficient Training",
            "Mixture of Experts",
            "Cross-Lingual Transfer",
            "Multilingual Pretraining",
            "Machine Translation",
            "Low-Resource NLP",
            "Language Modeling",
            "Named Entity Recognition",
            "Question Answering",
            "Text Classification",
            "Sentiment Analysis",
            "Summarization",
            "Information Extraction",
            "Speech and Audio",
            "Benchmark and Evaluation",
            "Other",
        ]
    ] = Field(..., description="Primary topic(s) or techniques in the paper.")
    subject_areas: List[
        Literal[
            "Computational Social Science and Cultural Analytics",
            "Machine Translation",
            "Multilingualism and Cross-Lingual NLP",
            "Multimodality and Language Grounding to Vision, Robotics and Beyond",
            "Question Answering",
            "Sentiment Analysis, Stylistic Analysis, and Argument Mining",
            "Speech Recognition, Text-to-Speech and Spoken Language Understanding",
            "Summarization",
            "Language Modeling",
            "Information Retrieval and Text Mining",
            "Information Extraction",
            "Generation",
            "Efficient Methods",
            "Resources and Evaluation",
        ]
    ] = Field(
        ...,
        description="Primary subject area of the paper based on ACL 2025 Subject Areas.",
    )
    modalities: List[
        Literal[
            "Text",
            "Speech",
            "Vision",
            "Multimodal",
            "Other",
        ]
    ] = Field(..., description="Modality of the work.")
    languages_supported: List[str] = Field(
        ...,
        description="Languages studied or supported. Use ISO 639-1 codes (e.g., en, fr, de). Use 'multilingual' if >10 languages.",
    )
    models_released: List[str] = Field(
        ..., description="Names of any models released by the authors. Empty list if none."
    )
    model_sizes: List[float] = Field(
        ...,
        description="Parameter sizes of models released in billions (e.g., 7, 1.5, 0.1). Empty list if none or not specified.",
    )
    relevance_score: int = Field(
        ...,
        description=(
            "Relevance to 'the last mile of NLP': the intersection of multilinguality and edge-efficiency, "
            "i.e., reaching language communities that need both a model that understands their language AND "
            "the efficiency to run on constrained hardware. Scored 1-5: "
            "1 - Not relevant: addresses neither multilinguality nor efficiency. "
            "2 - Slightly relevant: touches on one aspect peripherally. "
            "3 - Moderately relevant: substantially addresses multilinguality OR efficiency, but not both. "
            "4 - Very relevant: studies the tension between multilinguality and efficiency (e.g., how compression "
            "degrades multilingual performance, or how to maintain language coverage in smaller models). "
            "5 - Highly relevant: directly targets multilingual NLP on resource-constrained or edge devices, "
            "addressing the 'low-resource double bind' of limited data AND limited compute."
        ),
        ge=1,
        le=5,
    )
    relevance_reasoning: str = Field(
        ...,
        description="Brief reasoning for the assigned relevance score.",
    )
