document.addEventListener("DOMContentLoaded", () => {
  const pipelineMount = document.getElementById("pipeline-figure");
  if (pipelineMount && window.PipelineModule) {
    window.PipelineModule.renderPipeline(pipelineMount);
    window.PipelineModule.resetDetail();
  }

  // Figure scripts wire interactive analysis mounts independently.

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

  const drawer = document.getElementById("methodology");
  if (drawer) {
    const toggles = document.querySelectorAll("[data-drawer-open]");
    let lastFocus = null;
    const setOpen = (open) => {
      document.body.classList.toggle("drawer-open", open);
      toggles.forEach((t) => t.setAttribute("aria-expanded", String(open)));
      if (open) {
        lastFocus = document.activeElement;
        drawer.focus();
      } else if (lastFocus) {
        lastFocus.focus();
      }
    };
    toggles.forEach((t) => t.addEventListener("click", (e) => {
      e.preventDefault();
      setOpen(true);
    }));
    document.querySelectorAll("[data-drawer-close]").forEach((c) =>
      c.addEventListener("click", () => setOpen(false)));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && document.body.classList.contains("drawer-open")) setOpen(false);
    });
  }
});
