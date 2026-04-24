document.addEventListener("DOMContentLoaded", () => {
  const pipelineMount = document.getElementById("pipeline-figure");
  if (pipelineMount && window.PipelineModule) {
    window.PipelineModule.renderPipeline(pipelineMount);
    window.PipelineModule.resetDetail();
  }

  // Analysis tabs — click a tab to swap the visible panel.
  const tabs = document.querySelectorAll(".tabs .tab");
  const panels = document.querySelectorAll(".tab-panels .tab-panel");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const id = tab.dataset.tab;
      tabs.forEach((t) => {
        const active = t === tab;
        t.classList.toggle("is-active", active);
        t.setAttribute("aria-selected", active ? "true" : "false");
      });
      panels.forEach((p) => {
        const active = p.dataset.panel === id;
        p.classList.toggle("is-active", active);
        if (active) p.removeAttribute("hidden");
        else p.setAttribute("hidden", "");
      });
    });
  });

  // Chart mounts (#fig-how, #fig-who, #fig-where) stay as placeholders
  // until wired to Plotly/cytoscape in a follow-up.
});
