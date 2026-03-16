from typing import List, Literal

from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are an expert research paper annotator specializing in NLP and machine learning.
Given the title and abstract of a research paper, classify it according to several dimensions
relevant to multilingual and efficient NLP for edge deployment.
Your responses must strictly adhere to the specified response schema without adding any additional commentary."""

USER_PROMPT = """Title: {title}

Abstract: {abstract}

Please classify this paper according to the following dimensions:
1. The single primary pipeline stage the paper addresses (Data Collection, Pretraining, Post-training, Inference, Evaluation, or Full-Stack if it spans 4+ stages).
2. Primary topic(s) or techniques used in the paper.
3. Primary subject area of the paper based on ACL 2025 Subject Areas.
4. Modality of the work (e.g., text, speech, multimodal).
5. Languages studied or supported (if applicable). Use ISO 639-1 codes where possible (e.g., en, fr, de, es). Use "multilingual" if >10 languages.
6. Names of any models released by the authors (empty list if none).
7. Parameter sizes of the models released in billions (empty list if none or not specified).
8. Is this paper primarily about efficiency, multilinguality, both, or neither?
9. What type of contribution does this paper make (Method, Technique, Evaluation, Survey, Resource, Analysis)?
10. Is this paper relevant in the context of multilingual NLP for edge devices? Score from 1 to 5.
11. State your reason for the relevance score.
12. Extract a list of free-form keywords that capture the paper's key concepts, methods, datasets, or findings.
"""


class ResearchPaperAnnotation(BaseModel):
    pipeline_stage: Literal[
        "Data Collection",
        "Pretraining",
        "Post-training",
        "Inference",
        "Evaluation",
        "Full-Stack",
    ] = Field(
        ...,
        description=(
            "The single primary pipeline stage this paper addresses. Pick ONE: "
            "Data Collection: corpus creation, data augmentation, annotation. "
            "Pretraining: training from scratch, continued pretraining, foundation models. "
            "Post-training: fine-tuning, instruction tuning, alignment, adapters, LoRA. "
            "Inference: quantization, pruning, distillation, deployment, on-device. "
            "Evaluation: primarily a benchmark, metric, leaderboard, or analysis paper. "
            "Full-Stack: spans 4 or more stages (e.g., a complete model report covering data, pretraining, post-training, and inference)."
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
    research_focus: Literal[
        "Efficiency",
        "Multilinguality",
        "Both",
        "Neither",
    ] = Field(
        ...,
        description=(
            "Is this paper primarily about efficiency, multilinguality, both, or neither? "
            "Efficiency: compression, quantization, pruning, distillation, on-device, edge deployment, small models. "
            "Multilinguality: cross-lingual transfer, multilingual models, low-resource languages, translation. "
            "Both: explicitly studies the intersection (e.g., compressing multilingual models, efficient cross-lingual transfer). "
            "Neither: does not primarily focus on either dimension."
        ),
    )
    contribution_type: List[
        Literal[
            "Method",
            "Technique",
            "Evaluation",
            "Survey",
            "Resource",
            "Analysis",
        ]
    ] = Field(
        ...,
        description=(
            "What type of contribution does this paper make? "
            "Method: proposes a new model, architecture, or training procedure. "
            "Technique: introduces a specific trick, algorithm, or optimization (e.g., a new pruning strategy). "
            "Evaluation: primarily benchmarks or compares existing methods. "
            "Survey: reviews and summarizes existing literature. "
            "Resource: releases a dataset, benchmark, or tool. "
            "Analysis: provides empirical analysis or insights without proposing a new method."
        ),
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
    keywords: List[str] = Field(
        ...,
        description=(
            "Free-form keywords capturing the paper's key concepts, methods, datasets, or findings. "
            "Include specific technique names (e.g., 'GPTQ', 'LoRA'), dataset names (e.g., 'FLORES', 'XNLI'), "
            "model names, and other salient terms useful for building a keyword co-occurrence network."
        ),
    )
