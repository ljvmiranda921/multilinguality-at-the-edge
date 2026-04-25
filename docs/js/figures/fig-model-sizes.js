(function () {
  const MOUNT_ID = "fig-2b";
  const DATA_URL = "assets/data/model_sizes.json";
  const META_URL = "assets/data/model_sizes_meta.json";

  const COLORS = {
    ink:        "#35302E",
    accent:     "#254EFF",
    warm:       "#C96A2E",
    warm_pale:  "#F5D8BE",
    warm_light: "#E39A5F",
    cool:       "#7B93B8",
    rule:       "#e5ddcb",
    muted:      "#6b6358",
  };

  const TICK_VALS  = [0.3, 1, 3, 7, 14, 30, 70, 130, 400];
  const X_DOMAIN   = [0.2, 600];
  const ROW_HEIGHT = 14;
  const MARGINS    = { left: 138, right: 60, top: 32, bottom: 46 };
  const FONT_SIZE_NAME  = 11.5;
  const FONT_SIZE_TICK  = 10.5;
  const FONT_SIZE_LABEL = 12.5;
  const FONT_SIZE_CAT   = 13;
  const SVG_NS     = "http://www.w3.org/2000/svg";

  function logScale(domain, range) {
    const d0 = Math.log10(domain[0]);
    const d1 = Math.log10(domain[1]);
    const r0 = range[0];
    const r1 = range[1];
    return v => r0 + ((Math.log10(v) - d0) / (d1 - d0)) * (r1 - r0);
  }

  function el(name, attrs, children) {
    const e = document.createElementNS(SVG_NS, name);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (children) for (const c of children) {
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return e;
  }

  function buildSVG(records, width) {
    const innerW = width - MARGINS.left - MARGINS.right;
    const innerH = records.length * ROW_HEIGHT;
    const height = innerH + MARGINS.top + MARGINS.bottom;
    const xScale = logScale(X_DOMAIN, [0, innerW]);

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      width: width,
      height: height,
      style: "display: block; max-width: 100%; height: auto;",
    });

    // Category bands
    const cats = [
      { label: "Small",  x0: 0.2,  x1: 8,   color: COLORS.accent },
      { label: "Medium", x0: 8,    x1: 80,  color: COLORS.cool },
      { label: "Large",  x0: 80,   x1: 600, color: COLORS.warm },
    ];
    cats.forEach(cat => {
      const x0 = MARGINS.left + xScale(cat.x0);
      const x1 = MARGINS.left + xScale(cat.x1);
      svg.appendChild(el("rect", {
        x: x0, y: MARGINS.top, width: x1 - x0, height: innerH,
        fill: cat.color, opacity: 0.07,
      }));
      svg.appendChild(el("text", {
        x: (x0 + x1) / 2, y: MARGINS.top - 12,
        "text-anchor": "middle",
        "font-family": "Tomato Grotesk, sans-serif",
        "font-size": FONT_SIZE_CAT, "font-weight": 600,
        fill: cat.color,
      }, [cat.label]));
    });

    // Year separators
    let prevYear = null;
    records.forEach((r, i) => {
      if (prevYear !== null && r.year !== prevYear) {
        const y = MARGINS.top + i * ROW_HEIGHT;
        svg.appendChild(el("line", {
          x1: 0, x2: width, y1: y, y2: y,
          stroke: COLORS.rule, "stroke-width": 1,
        }));
      }
      prevYear = r.year;
    });

    // Year labels (centered per year group)
    const yearGroups = {};
    records.forEach((r, i) => {
      (yearGroups[r.year] = yearGroups[r.year] || []).push(i);
    });
    Object.entries(yearGroups).forEach(([year, indices]) => {
      const midI = indices.reduce((a, b) => a + b, 0) / indices.length;
      const y = MARGINS.top + (midI + 0.5) * ROW_HEIGHT;
      svg.appendChild(el("text", {
        x: width - MARGINS.right + 10,
        y: y,
        "text-anchor": "start",
        "dominant-baseline": "middle",
        "font-family": "Univers, sans-serif",
        "font-size": FONT_SIZE_TICK,
        "font-style": "italic",
        fill: COLORS.muted,
      }, [year]));
    });

    // Rows
    records.forEach((r, i) => {
      const y = MARGINS.top + (i + 0.5) * ROW_HEIGHT;

      // Model name
      const nameG = el("g", {
        class: "model-name",
        "data-model": r.name,
      });
      nameG.appendChild(el("text", {
        x: MARGINS.left - 10, y: y,
        "text-anchor": "end",
        "dominant-baseline": "middle",
        "font-family": "Univers, sans-serif",
        "font-size": FONT_SIZE_NAME,
        fill: COLORS.ink,
      }, [r.name]));
      svg.appendChild(nameG);

      // Range line
      if (r.sizes.length > 1) {
        svg.appendChild(el("line", {
          x1: MARGINS.left + xScale(r.min),
          x2: MARGINS.left + xScale(r.max),
          y1: y, y2: y,
          stroke: COLORS.accent,
          "stroke-width": 2,
          "stroke-linecap": "round",
        }));
      }

      // Circles
      r.sizes.forEach(s => {
        svg.appendChild(el("circle", {
          cx: MARGINS.left + xScale(s),
          cy: y, r: 4.5,
          fill: COLORS.warm_pale,
          stroke: COLORS.warm,
          "stroke-width": 1.4,
          class: "model-dot",
          "data-model": r.name,
          "data-size": String(s),
        }));
      });
    });

    // X axis
    const axisY = MARGINS.top + innerH + 8;
    svg.appendChild(el("line", {
      x1: MARGINS.left, x2: MARGINS.left + innerW,
      y1: axisY, y2: axisY,
      stroke: COLORS.ink, "stroke-width": 1,
    }));
    TICK_VALS.forEach(t => {
      const tx = MARGINS.left + xScale(t);
      svg.appendChild(el("line", {
        x1: tx, x2: tx, y1: axisY, y2: axisY + 4,
        stroke: COLORS.ink, "stroke-width": 1,
      }));
      svg.appendChild(el("text", {
        x: tx, y: axisY + 15,
        "text-anchor": "middle",
        "font-family": "Univers, sans-serif",
        "font-size": FONT_SIZE_TICK,
        fill: COLORS.ink,
      }, [String(t)]));
    });

    svg.appendChild(el("text", {
      x: MARGINS.left + innerW / 2, y: axisY + 33,
      "text-anchor": "middle",
      "font-family": "Tomato Grotesk, sans-serif",
      "font-size": FONT_SIZE_LABEL,
      fill: COLORS.ink,
    }, ["Model size (B parameters)"]));

    return svg;
  }

  // -------- tooltip --------

  let tooltipEl = null;
  let hideTimer = null;

  function ensureTooltip() {
    if (tooltipEl) return tooltipEl;
    tooltipEl = document.createElement("div");
    tooltipEl.className = "model-tooltip";
    tooltipEl.setAttribute("role", "tooltip");
    tooltipEl.addEventListener("mouseenter", () => clearTimeout(hideTimer));
    tooltipEl.addEventListener("mouseleave", () => scheduleHide());
    document.body.appendChild(tooltipEl);
    return tooltipEl;
  }

  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      if (tooltipEl) tooltipEl.classList.remove("is-visible");
    }, 180);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;",
    })[c]);
  }

  function showTooltip(name, meta, evt) {
    const t = ensureTooltip();
    const desc = meta?.description?.trim()
      ? escapeHtml(meta.description)
      : '<span class="muted">No description yet.</span>';
    const link = meta?.tech_report_url
      ? ` <a href="${escapeHtml(meta.tech_report_url)}" target="_blank" rel="noopener">tech report &rarr;</a>`
      : "";
    const hf = meta?.hf_url
      ? ` <a href="${escapeHtml(meta.hf_url)}" target="_blank" rel="noopener">HF &rarr;</a>`
      : "";
    t.innerHTML =
      `<div class="model-tooltip-name">${escapeHtml(name)}</div>` +
      `<div class="model-tooltip-desc">${desc}</div>` +
      (link || hf
        ? `<div class="model-tooltip-links">${link}${hf}</div>`
        : "");
    t.classList.add("is-visible");
    positionTooltip(evt);
    clearTimeout(hideTimer);
  }

  function positionTooltip(evt) {
    if (!tooltipEl) return;
    const pad = 14;
    const tw = tooltipEl.offsetWidth;
    const th = tooltipEl.offsetHeight;
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    if (x + tw > window.innerWidth - 8)  x = evt.clientX - tw - pad;
    if (y + th > window.innerHeight - 8) y = evt.clientY - th - pad;
    tooltipEl.style.left = x + "px";
    tooltipEl.style.top  = y + "px";
  }

  // -------- interactions --------

  function wire(mount, meta) {
    mount.querySelectorAll(".model-name").forEach(g => {
      const name = g.dataset.model;
      g.style.cursor = "pointer";
      g.addEventListener("mouseenter", e => showTooltip(name, meta[name], e));
      g.addEventListener("mousemove",  e => positionTooltip(e));
      g.addEventListener("mouseleave", () => scheduleHide());
      g.addEventListener("click", () => {
        const url = meta[name]?.tech_report_url || meta[name]?.hf_url;
        if (url) window.open(url, "_blank", "noopener");
      });
    });

    mount.querySelectorAll(".model-dot").forEach(c => {
      const name = c.dataset.model;
      const size = c.dataset.size;
      const m    = meta[name] || {};
      const url  = (m.hf_urls && m.hf_urls[size]) || m.hf_url;
      if (url) {
        c.style.cursor = "pointer";
        c.addEventListener("click", () => window.open(url, "_blank", "noopener"));
      }
      c.addEventListener("mouseenter", () => {
        c.setAttribute("fill", COLORS.warm_light);
      });
      c.addEventListener("mouseleave", () => {
        c.setAttribute("fill", COLORS.warm_pale);
      });
    });
  }

  function render(mount, data, meta) {
    mount.innerHTML = "";
    const width = Math.max(420, Math.floor(mount.clientWidth || mount.getBoundingClientRect().width));
    const svg = buildSVG(data.models, width);
    mount.appendChild(svg);
    wire(mount, meta);
  }

  async function boot() {
    const mount = document.getElementById(MOUNT_ID);
    if (!mount) return;
    try {
      const [dataRes, metaRes] = await Promise.all([
        fetch(DATA_URL, { cache: "no-cache" }),
        fetch(META_URL, { cache: "no-cache" }),
      ]);
      if (!dataRes.ok) throw new Error(`data: HTTP ${dataRes.status}`);
      const data = await dataRes.json();
      const meta = metaRes.ok ? await metaRes.json() : {};
      render(mount, data, meta);

      let resizeTimer = null;
      window.addEventListener("resize", () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => render(mount, data, meta), 120);
      });
    } catch (err) {
      console.error("fig-model-sizes:", err);
      mount.innerHTML = '<p class="placeholder">[ figure failed to load ]</p>';
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
