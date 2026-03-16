from typing import List, Literal

from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are an expert research paper annotator. Given the title and abstract of a research paper, you will classify it according to several dimensions relevant to low-resource and efficient NLP on edge devices. Your responses must strictly adhere to the specified response schema without adding any additional commentary or information."""
USER_PROMPT = """Title: {title}\n\nAbstract: {abstract}\n\nPlease classify the paper according to the following dimensions:
1. Primary subject area of the paper based on ACL 2025 Subject Areas.
2. Application domain(s) addressed by the paper.
3. Methods used in the paper.
4. Deployment platforms targeted by the paper.
5. Names of any models released by the authors.
6. Parameter sizes of the models released (in B of parameters).
7. Modality of the models (e.g., text, speech, multimodal).
8. Languages supported by the models (if applicable). Use the ISO 639-1 codes where possible (e.g., en, fr, de, es).
9. Is this paper relevant in the context of efficient NLP on edge devices (score from 1 to 5, with 5 as highly relevant)?
10. State your reason as to why you assigned the relevance score.
"""


class ResearchPaperAnnotation(BaseModel):
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
        ]
    ] = Field(
        ...,
        description="Primary subject area of the paper based on ACL 2025 Subject Areas.",
    )
    application_domain: List[
        Literal[
            "Healthcare",
            "Education",
            "Finance",
            "Agriculture",
            "Legal",
            "Smart Cities",
            "Industrial IoT",
            "Environmental Science",
            "Other",
        ]
    ] = Field(..., description="Application domain(s) addressed by the paper.")
    methods_used: List[
        Literal[
            "Quantization",
            "Pruning",
            "Knowledge Distillation",
            "Neural Architecture Search",
            "Low-Rank Factorization",
            "Federated Learning",
            "Sparse Modeling",
            "Parameter-Efficient Fine-Tuning",
            "Efficient Architectures",
            "Data-Efficient Training",
            "Mixture of Experts",
            "Other",
        ]
    ] = Field(..., description="Methods used in the paper.")
    deployment_platforms: List[
        Literal[
            "Microcontrollers",
            "Wearables",
            "Mobile Devices",
            "Laptop/PC",
            "Cloud",
            "Not Specified",
        ]
    ] = Field(..., description="Deployment platforms targeted by the paper.")
    models_released: List[str] = Field(
        ..., description="Names of any models released by the authors."
    )
    model_sizes: List[float] = Field(
        ...,
        description="Parameter sizes of the models released (in billions of parameters, e.g., 405, 70, 8, 1, 0.2).",
    )
    modalities: List[
        Literal[
            "Text",
            "Speech",
            "Vision",
            "Multimodal",
            "Other",
        ]
    ] = Field(
        ...,
        description="Modality of the models (e.g., text, speech, vision multimodal).",
    )
    languages_supported: List[str] = Field(
        ...,
        description="Languages supported by the models (if applicable). Use the ISO 639-1 codes where possible (e.g., en, fr, de, es).",
    )
    relevance_score: int = Field(
        ...,
        description="Is this paper relevant in the context of efficient NLP on edge devices. Score from 1 to 5 using the following rubrics:"
        "1 - Not relevant: The paper does not address any aspects of efficient NLP or edge devices.\n"
        "2 - Slightly relevant: The paper mentions efficient NLP or edge devices but does not focus on them.\n"
        "3 - Moderately relevant: The paper addresses efficient NLP or edge devices but lacks depth or novelty.\n"
        "4 - Very relevant: The paper provides significant contributions to efficient NLP by either introducing a novel method, a new benchmark, or demonstrating substantial improvements.\n"
        "5 - Highly relevant: The paper is an important and seminal work in efficient NLP on edge devices.",
        ge=1,
        le=5,
    )
    relevance_reasoning: str = Field(
        ...,
        description="Reasoning for the assigned relevance score.",
    )
