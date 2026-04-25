// ============================================================
// EDIT ME — pipeline stage copy
// Add the survey text per stage. Use [1], [2] markers inline;
// the numbers map to REFERENCES below (1-indexed).
// ============================================================

const PIPELINE_STAGES = {
  // Pipeline stages
  "data-collection": {
    title: "Data Collection",
    body: "The LM pipeline often begins with sourcing and curating text corpora for both pretraining and post-training. " + 
          "Pretraining corpora tend to be unstructured web-crawled text, while post-training datasets are more structured.",
  },
  "pretraining": {
    title: "Pretraining",
    body:  "TODO — paste survey text for the Pretraining stage here.",
  },
  "post-training": {
    title: "Post-training",
    body:  "TODO — paste survey text for the Post-training stage here.",
  },
  "inference": {
    title: "Inference",
    body:  "TODO — paste survey text for the Inference stage here.",
  },
  "evaluation": {
    title: "Evaluation",
    body:  "TODO — paste survey text for the Evaluation stage here.",
  },

  // Requirements of the edge (top band)
  "req-memory": {
    title: "Memory",
    body:  "TODO — paste survey text on memory constraints at the edge.",
  },
  "req-compute": {
    title: "Compute",
    body:  "TODO — paste survey text on compute constraints at the edge.",
  },
  "req-energy": {
    title: "Energy",
    body:  "TODO — paste survey text on energy constraints at the edge.",
  },

  // Requirements for multilingual capability (bottom band)
  "req-data": {
    title: "Data",
    body:  "TODO — paste survey text on data requirements for multilinguality.",
  },
  "req-representation": {
    title: "Representation",
    body:  "TODO — paste survey text on representation for multilinguality.",
  },
  "req-alignment": {
    title: "Alignment",
    body:  "TODO — paste survey text on alignment for multilinguality.",
  },
};

// ============================================================
// EDIT ME — numbered references (1-indexed as shown to readers)
// ============================================================

const REFERENCES = [
  "TODO — Author et al. (Year). Title. Venue.",
  "TODO — second reference",
];

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

  const body = linkifyRefs(escapeHtml(stage.body));
  const refsHtml = REFERENCES.length
    ? `<div class="refs"><strong>References</strong><ol>${
        REFERENCES.map((r, i) => `<li id="ref-${i + 1}">${escapeHtml(r)}</li>`).join("")
      }</ol></div>`
    : "";

  detail.innerHTML = `
    <h3>${escapeHtml(stage.title)}</h3>
    <p>${body}</p>
    ${refsHtml}
  `;
}

function linkifyRefs(text) {
  return text.replace(/\[(\d+)\]/g, (_, n) =>
    `<sup class="ref-marker"><a href="#ref-${n}">[${n}]</a></sup>`
  );
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// expose for main.js
window.PipelineModule = { renderPipeline, resetDetail };
