(function () {
  async function load(el) {
    const url = el.dataset.svg;
    if (!url) return;
    try {
      const res = await fetch(url, { cache: "no-cache" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      el.innerHTML = await res.text();
      el.classList.add("svg-loaded");
    } catch (err) {
      console.error("SVG load:", url, err);
      el.innerHTML = '<p class="placeholder">[ figure failed to load ]</p>';
    }
  }

  function boot() {
    document.querySelectorAll("[data-svg]").forEach(load);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
