# Annotation Guidelines: Research Direction

## Overview

Each paper in our survey touches on **efficiency** (edge/small models) and/or **multilinguality** (cross-lingual, low-resource).
We want to capture the **directionality of motivation**: which concern came first, and which is secondary?

## Categories

### `efficiency_first`

**Definition:** The paper's core contribution is an efficiency technique (quantization, pruning, distillation, compact architecture, etc.). Multilinguality appears as a downstream evaluation or secondary consideration.

**Key signal:** The method is language-agnostic. It would exist even if only English existed. Multilingual experiments are included to show generality, not because the method was designed for multilingual settings.

**Ask yourself:** "If I removed all the multilingual experiments, would the paper still make sense and have the same core contribution?"
- If **yes** → `efficiency_first`

**Examples:**
- SmoothQuant evaluated on mT5 → the quantization method is the contribution, multilingual is just one eval axis
- SparseGPT tested on multilingual benchmarks → pruning method is language-agnostic
- QLoRA applied to fine-tune a multilingual model → the efficient fine-tuning is the point, not the language coverage
- BitNet / GPTQ / AWQ → compression methods that happen to be tested on multilingual models

### `multilingual_first`

**Definition:** The paper's core contribution addresses a multilingual or cross-lingual problem. Efficiency techniques are used as practical tools to make the multilingual solution feasible or deployable.

**Key signal:** The method is designed *around* language-specific challenges (vocabulary design, script differences, cross-lingual transfer, low-resource data scarcity). Efficiency is a means to an end, not the end itself.

**Ask yourself:** "If I replaced the efficient method with an expensive one (e.g., full fine-tuning instead of LoRA, large model instead of small), would the paper's core research question remain the same?"
- If **yes** → `multilingual_first`

**Examples:**
- MAD-X using adapters for cross-lingual transfer → adapters are efficient, but the contribution is the cross-lingual framework
- MaLA-500 adapting LLaMA to 500 languages → the language adaptation strategy is the point, LoRA is just the tool
- Aya Model fine-tuning for multilingual instruction following → multilingual coverage is the goal, efficient tuning is the vehicle
- Glot500 scaling to 500 languages → the contribution is language coverage, not the training efficiency

### `co_designed`

**Definition:** The paper explicitly designs a method that *jointly* addresses efficiency and multilinguality. Neither concern is secondary — the contribution is specifically about their interaction.

**Key signal:** The paper studies or exploits the *tension* between efficiency and multilinguality. Removing either dimension would fundamentally change the paper.

**Ask yourself:** "Does this paper study how efficiency affects multilinguality (or vice versa), or propose a method that specifically addresses both?"
- If **yes** → `co_designed`

**Examples:**
- SLAM: Selective Language Alignment for efficient multilingual reasoning → explicitly designs alignment to be both language-aware and compute-efficient
- Phi-3 Technical Report → designed from the ground up to be small AND multilingual
- A study on how quantization disproportionately degrades low-resource languages → the interaction IS the contribution
- EmbeddingGemma → lightweight text representations designed for multilingual use

## Edge Cases and Decision Rules

1. **Full-Stack model reports** (Qwen2.5, Gemma 3, etc.): Usually `co_designed` if they explicitly discuss multilingual data curation AND model compression/efficiency. If multilingual coverage is just listed as a feature with no special treatment, lean `efficiency_first`.

2. **Benchmark papers** that evaluate efficient models on multilingual tasks: Usually `efficiency_first` unless the benchmark itself is designed to measure the efficiency-multilinguality tradeoff.

3. **Machine translation with efficient methods**: Usually `multilingual_first` — the core problem is translation (inherently multilingual), and efficiency is applied to make it practical.

4. **Adapters/LoRA for cross-lingual transfer**: Usually `multilingual_first` — PEFT is the tool, cross-lingual transfer is the goal.

5. **Adapters/LoRA as a general PEFT method** tested on multilingual benchmarks: Usually `efficiency_first` — the PEFT method is the contribution.

6. **Papers tagged `research_focus: Efficiency`**: Most will be `efficiency_first`, but not always (e.g., an efficient method specifically designed for low-resource languages).

7. **Papers tagged `research_focus: Multilinguality`**: Most will be `multilingual_first`, but not always (e.g., a multilingual benchmark that specifically tests model compression effects).

8. **Papers tagged `research_focus: Both`**: Could be any of the three. Read carefully — `Both` means the paper touches both themes, but `co_designed` means it *jointly optimizes* for them.
