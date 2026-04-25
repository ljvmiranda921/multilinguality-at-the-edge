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
    body: "The LM pipeline often begins with sourcing and curating text corpora for both pretraining and post-training. " + 
          "Pretraining corpora tend to be unstructured web-crawled text, while post-training datasets are more structured.",
    references: [],
  },
  "pretraining": {
    title: "Pretraining",
    body:  "TODO — paste survey text for the Pretraining stage here.",
    references: [],
  },
  "post-training": {
    title: "Post-training",
    body:  "TODO — paste survey text for the Post-training stage here.",
    references: [],
  },
  "inference": {
    title: "Inference",
    body:  "TODO — paste survey text for the Inference stage here.",
    references: [],
  },
  "evaluation": {
    title: "Evaluation",
    body:  "TODO — paste survey text for the Evaluation stage here.",
    references: [],
  },

  // Requirements of the edge (top band)
  "req-memory": {
    title: "Memory",
    body:  "TODO — paste survey text on memory constraints at the edge.",
    references: [],
  },
  "req-compute": {
    title: "Compute",
    body:  "TODO — paste survey text on compute constraints at the edge.",
    references: [],
  },
  "req-energy": {
    title: "Energy",
    body:  "TODO — paste survey text on energy constraints at the edge.",
    references: [],
  },

  // Requirements for multilingual capability (bottom band)
  "req-data": {
    title: "Data",
    body:  "TODO — paste survey text on data requirements for multilinguality.",
    references: [],
  },
  "req-representation": {
    title: "Representation",
    body:  "TODO — paste survey text on representation for multilinguality.",
    references: [],
  },
  "req-alignment": {
    title: "Alignment",
    body:  "TODO — paste survey text on alignment for multilinguality.",
    references: [],
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
