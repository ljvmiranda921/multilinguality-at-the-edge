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

  // Copy-to-clipboard buttons. A button with data-copy-target="#id"
  // copies the target element's textContent to the clipboard and
  // briefly shows a confirmation label.
  document.querySelectorAll(".copy-btn[data-copy-target]").forEach((btn) => {
    const originalText = btn.textContent;
    let resetTimer = null;
    btn.addEventListener("click", async () => {
      const target = document.querySelector(btn.dataset.copyTarget);
      if (!target) return;
      const text = target.textContent.trim();
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied to clipboard";
        btn.classList.add("is-copied");
      } catch (err) {
        btn.textContent = "Copy failed";
      }
      clearTimeout(resetTimer);
      resetTimer = setTimeout(() => {
        btn.textContent = originalText;
        btn.classList.remove("is-copied");
      }, 2000);
    });
  });
});
