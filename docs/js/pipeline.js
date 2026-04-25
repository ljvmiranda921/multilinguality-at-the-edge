// ============================================================
// EDIT ME — pipeline stage copy
// Add the survey text per stage/requirement. Use [1], [2] markers
// inline; numbering is LOCAL to that stage/requirement and maps to
// its own `references` array (1-indexed).
// ============================================================

const PIPELINE_STAGES = {
  // Pipeline stages
  "data-collection": {
    title: "Data Collection",
    body: "The LM pipeline begins by sourcing and curating corpora for both pretraining and post-training. A core challenge is maximizing language coverage under limited model capacity while avoiding multilingual noise that disproportionately affects low-resource languages [1][2]. Common methods include language-mixture design, synthetic data generation, and strong multilingual filtering/curation pipelines [3][4].",
    references: [
      "Doddapaneni et al. (2025). A Primer on Pretrained Multilingual Language Models.",
      "Chang et al. (2024). When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages.",
      "Dang et al. (2024). Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier.",
      "Kudugunta et al. (2023). MADLAD-400: A Multilingual And Document-Level Large Audited Dataset.",
    ],
  },
  "pretraining": {
    title: "Pretraining",
    body:  "Pretraining is usually the most compute-intensive stage and sets much of the multilingual ceiling for downstream performance. The key tension is that multilingual quality tends to improve with scale, but edge deployment imposes strict memory and compute limits [1]. Practical methods include better tokenizer/vocabulary design, continual pretraining with vocabulary expansion, and sparse/expert-style architectures that activate fewer parameters per input [2][3][4].",
    references: [
      "Longpre et al. (2025). ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding.",
      "Petrov et al. (2023). Language Model Tokenizers Introduce Unfairness Between Languages.",
      "Kim et al. (2024). Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models.",
      "Muennighoff et al. (2025). OLMoE: Open Mixture-of-Experts Language Models.",
    ],
  },
  "post-training": {
    title: "Post-training",
    body:  "Post-training adapts a base LM via SFT/RL and is where most communities begin practical model development. A recurring challenge is that useful base models are often too large for target edge devices, and adding/removing capabilities through retraining is expensive [1]. Methods include quantization/pruning, distillation, model merging, and federated adaptation to reduce memory/compute costs while preserving multilingual behavior [2][3][4].",
    references: [
      "Zheng et al. (2025). A Review on Edge Large Language Models: Design, Execution, and Applications.",
      "Lin et al. (2025). AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.",
      "Agarwal et al. (2024). On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.",
      "Yadav et al. (2023). TIES-Merging: Resolving Interference When Merging Models.",
      "Li et al. (2025). Multilingual Federated Low-Rank Adaptation for Collaborative Content Anomaly Detection.",
    ],
  },
  "inference": {
    title: "Inference",
    body:  "Inference is where edge constraints are most visible: online serving depends on connectivity, while offline serving depends on local memory and battery. A key challenge is that inference cost is language-dependent because tokenization efficiency varies by language, creating a practical token tax [1][2]. Common methods are prompt compression, speculative decoding, and inference-time tokenization/vocabulary adaptation for target languages [3][4].",
    references: [
      "Fu et al. (2025). LLMCO2: Advancing Accurate Carbon Footprint Prediction for LLM Inferences.",
      "Lundin et al. (2026). The Token Tax: Systematic Bias in Multilingual Tokenization.",
      "Jiang et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.",
      "Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding.",
    ],
  },
  "evaluation": {
    title: "Evaluation",
    body:  "Evaluation guides model selection for deployment and can dominate development cost in multilingual settings. The challenge is that broad multilingual benchmarks are expensive to run and quickly become stale as models improve [1][2]. Methods include lite benchmark variants, informative instance selection, and dynamic/refreshable evaluation pipelines [3][4].",
    references: [
      "Singh et al. (2025). Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation.",
      "Ojo et al. (2025). AfroBench: How Good are Large Language Models on African Languages?",
      "Micallef et al. (2025). MELABenchv1: Benchmarking LLMs for Low-Resource and Dialectal Maltese.",
      "Kim et al. (2025). BenchHub: A Unified Benchmark Suite for Holistic and Customizable LLM Evaluation.",
    ],
  },

  // Requirements of the edge (top band)
  "req-memory": {
    title: "Memory",
    body:  "Memory limits determine whether a model can even be loaded, and they also constrain fine-tuning batch sizes on-device. This creates a hard tradeoff between multilingual coverage and deployability on commodity hardware [1]. Typical mitigations are lower-precision quantization, compact model design, and memory-aware adaptation strategies [2].",
    references: [
      "Zheng et al. (2025). A Review on Edge Large Language Models: Design, Execution, and Applications.",
      "Lin et al. (2025). AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.",
    ],
  },
  "req-compute": {
    title: "Compute",
    body:  "Compute budget governs latency and practical feasibility of adaptation/inference on edge devices. Multilingual modeling adds complexity beyond task/domain variation, so dense architectures can become prohibitively expensive in low-FLOPS environments [1]. Methods that reduce active compute include sparse or expert-routed architectures and parameter-efficient adaptation [2][3].",
    references: [
      "Zheng et al. (2025). A Review on Edge Large Language Models: Design, Execution, and Applications.",
      "Muennighoff et al. (2025). OLMoE: Open Mixture-of-Experts Language Models.",
      "Li et al. (2025). Multilingual Federated Low-Rank Adaptation for Collaborative Content Anomaly Detection.",
    ],
  },
  "req-energy": {
    title: "Energy",
    body:  "Energy is a first-class deployment constraint because sustained generation can rapidly drain batteries on mobile and embedded devices. In multilingual settings, inefficient tokenization can increase tokens per query and therefore energy per output [1][2]. Mitigations include quantized/efficient inference stacks and reducing token counts through prompt or decoding optimizations [3].",
    references: [
      "Luccioni et al. (2024). Power Hungry Processing: Watts Driving the Cost of AI Deployment?",
      "Ahia et al. (2023). Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.",
      "Jiang et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.",
    ],
  },

  // Requirements for multilingual capability (bottom band)
  "req-data": {
    title: "Data",
    body:  "Multilingual capability depends on high-quality and diverse data, especially for low-resource communities. The challenge is obtaining broad coverage without introducing heavy noise or bias that harms smaller languages [1]. Representative approaches combine curated web crawling, community/annotated instruction data, and synthetic generation with quality control [2][3][4].",
    references: [
      "Penedo et al. (2025). FineWeb2: One Pipeline to Scale Them All.",
      "Singh et al. (2024). Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning.",
      "Dang et al. (2024). Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier.",
      "Kudugunta et al. (2023). MADLAD-400: A Multilingual And Document-Level Large Audited Dataset.",
    ],
  },
  "req-representation": {
    title: "Representation",
    body:  "Representation choices (tokenizer, vocabulary allocation, and encoding granularity) strongly shape multilingual model quality. The challenge is that a single representation often advantages some scripts/morphologies while penalizing others [1][2]. Methods include multilingual tokenizer redesign, byte-level modeling, and stronger cross-lingual pretraining setups [3][4].",
    references: [
      "Rust et al. (2021). How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.",
      "Petrov et al. (2023). Language Model Tokenizers Introduce Unfairness Between Languages.",
      "Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale.",
      "Minixhofer et al. (2026). Bolmo: Byteifying the Next Generation of Language Models.",
    ],
  },
  "req-alignment": {
    title: "Alignment",
    body:  "Fluency alone does not guarantee cultural appropriateness, safety, or normative alignment across communities. The challenge is that values and safety expectations vary by language and region, so one-size-fits-all alignment can fail [1][2]. Current directions emphasize culturally aware evaluation and multilingual safety/alignment interventions tailored to local contexts [3].",
    references: [
      "Adilazuarda et al. (2024). Towards Measuring and Modeling \"Culture\" in LLMs: A Survey.",
      "Liu et al. (2025). Culturally Aware and Adapted NLP: A Taxonomy and Survey of the State of the Art.",
      "Yong et al. (2025). The State of Multilingual LLM Safety Research.",
    ],
  },
};

// ============================================================
// Below this line: rendering + interaction. No need to edit
// for content updates.
// ============================================================

const STAGE_ORDER = [
  "data-collection",
  "pretraining",
  "post-training",
  "inference",
  "evaluation",
];

const TOP_BAND = ["req-memory", "req-compute", "req-energy"];
const BOTTOM_BAND = ["req-data", "req-representation", "req-alignment"];

function renderPipeline(mount) {
  mount.innerHTML = "";

  const stack = document.createElement("div");
  stack.className = "pipeline-stack";

  stack.appendChild(buildBand("Requirements of the Edge", TOP_BAND, "top"));
  stack.appendChild(buildStrip());
  stack.appendChild(buildBand("Requirements for Multilingual Capability", BOTTOM_BAND, "bottom"));

  mount.appendChild(stack);
}

function buildBand(label, ids, pos) {
  const wrap = document.createElement("div");
  wrap.className = `pipeline-band-wrap pipeline-band-${pos}`;

  const labelEl = document.createElement("div");
  labelEl.className = "pipeline-band";
  labelEl.textContent = label;

  const row = document.createElement("div");
  row.className = "pipeline-reqs";
  ids.forEach((id) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pipeline-req";
    btn.dataset.stage = id;
    btn.textContent = PIPELINE_STAGES[id].title;
    btn.addEventListener("click", () => toggleStage(id));
    row.appendChild(btn);
  });

  wrap.appendChild(labelEl);
  wrap.appendChild(row);
  return wrap;
}

function buildStrip() {
  const strip = document.createElement("div");
  strip.className = "pipeline-strip";
  STAGE_ORDER.forEach((id) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pipeline-segment";
    btn.dataset.stage = id;
    btn.textContent = PIPELINE_STAGES[id].title;
    btn.addEventListener("click", () => toggleStage(id));
    strip.appendChild(btn);
  });
  return strip;
}

let openStage = null;

function toggleStage(id) {
  if (openStage === id) {
    openStage = null;
    resetDetail();
  } else {
    openStage = id;
    renderDetail(id);
  }
  document.querySelectorAll(".pipeline-segment, .pipeline-req").forEach((el) => {
    el.classList.toggle("is-active", el.dataset.stage === openStage);
  });
}

function resetDetail() {
  const detail = document.getElementById("pipeline-detail");
  detail.innerHTML = '<p class="placeholder">[ click a stage above to read ]</p>';
}

function renderDetail(id) {
  const stage = PIPELINE_STAGES[id];
  const detail = document.getElementById("pipeline-detail");
  const references = Array.isArray(stage.references) ? stage.references : [];

  const body = linkifyRefs(escapeHtml(stage.body), id, references.length);
  const refsHtml = references.length
    ? `<div class="refs"><strong>References</strong><ol>${
        references.map((r, i) => `<li id="ref-${id}-${i + 1}">${escapeHtml(r)}</li>`).join("")
      }</ol></div>`
    : "";

  detail.innerHTML = `
    <h3>${escapeHtml(stage.title)}</h3>
    <p>${body}</p>
    ${refsHtml}
  `;
}

function linkifyRefs(text, stageId, refCount) {
  return text.replace(/\[(\d+)\]/g, (_, nRaw) => {
    const n = Number(nRaw);
    if (!Number.isFinite(n) || n < 1 || n > refCount) return `[${nRaw}]`;
    return `<sup class="ref-marker"><a href="#ref-${stageId}-${n}">[${n}]</a></sup>`;
  });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// expose for main.js
window.PipelineModule = { renderPipeline, resetDetail };
