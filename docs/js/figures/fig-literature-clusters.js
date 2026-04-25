(function () {
  const MOUNT_ID = "fig-how";
  const DATA_URL = "assets/data/literature_clusters.json";

  // 12 distinct hues — picked to read on the warm paper background.
  const CLUSTER_PALETTE = [
    "#254EFF", "#C96A2E", "#7B93B8", "#A368DF",
    "#4DB78C", "#CD3572", "#00BDB6", "#FD8153",
    "#5366E0", "#681FB1", "#E18AAC", "#13553A",
  ];
  const NOISE_COLOR = "#B5BDC8";
  const INK         = "#35302E";
  const MUTED       = "#6b6358";

  const SVG_NS  = "http://www.w3.org/2000/svg";
  const MARGIN  = { left: 20, right: 20, top: 20, bottom: 60 };

  function el(name, attrs, children) {
    const e = document.createElementNS(SVG_NS, name);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (children) for (const c of children) {
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return e;
  }

  function starPath(cx, cy, r) {
    const pts = [];
    for (let i = 0; i < 10; i++) {
      const angle  = -Math.PI / 2 + (i * Math.PI) / 5;
      const radius = i % 2 === 0 ? r : r * 0.42;
      pts.push([cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)]);
    }
    return "M" + pts.map(p => p.join(",")).join(" L") + " Z";
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;",
    })[c]);
  }

  // -------- tooltip --------

  let tipEl = null;
  let hideTimer = null;
  let activeCluster = null;
  let svgRoot = null;

  function ensureTip() {
    if (tipEl) return tipEl;
    tipEl = document.createElement("div");
    tipEl.className = "cluster-tooltip";
    tipEl.setAttribute("role", "tooltip");
    tipEl.addEventListener("mouseenter", () => clearTimeout(hideTimer));
    tipEl.addEventListener("mouseleave", () => scheduleClear());
    document.body.appendChild(tipEl);
    return tipEl;
  }

  function showCluster(cluster, color, evt) {
    clearTimeout(hideTimer);
    if (activeCluster === cluster.id) {
      positionTip(evt);
      return;
    }
    activeCluster = cluster.id;
    if (svgRoot) {
      svgRoot.classList.add("has-active");
      svgRoot
        .querySelectorAll('[data-cluster]')
        .forEach(node => {
          const same = node.getAttribute("data-cluster") === String(cluster.id);
          node.classList.toggle("is-active", same);
        });
    }

    const t = ensureTip();
    const reps = (cluster.representatives || [])
      .map(r => {
        const year = r.year ? ` (${r.year})` : "";
        const titleHtml = escapeHtml(r.title) + escapeHtml(year);
        return r.url
          ? `<li><a href="${escapeHtml(r.url)}" target="_blank" rel="noopener">${titleHtml}</a></li>`
          : `<li>${titleHtml}</li>`;
      })
      .join("");
    const kwsHtml = (cluster.keywords || [])
      .slice(0, 4)
      .map(k => `<span class="cluster-tooltip-kw">${escapeHtml(k)}</span>`)
      .join("");

    t.innerHTML =
      `<div class="cluster-tooltip-head">` +
        `<span class="cluster-tooltip-dot" style="background:${color}"></span>` +
        `<span class="cluster-tooltip-count">cluster of ${cluster.size}</span>` +
      `</div>` +
      `<div class="cluster-tooltip-kws">${kwsHtml}</div>` +
      `<div class="cluster-tooltip-reps-label">representative papers</div>` +
      `<ul class="cluster-tooltip-reps">${reps}</ul>`;
    t.style.borderColor = color;
    t.classList.add("is-visible");
    positionTip(evt);
  }

  function scheduleClear() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(clearActive, 220);
  }

  function clearActive() {
    activeCluster = null;
    if (svgRoot) {
      svgRoot.classList.remove("has-active");
      svgRoot
        .querySelectorAll('[data-cluster]')
        .forEach(n => n.classList.remove("is-active"));
    }
    if (tipEl) tipEl.classList.remove("is-visible");
  }

  function positionTip(evt) {
    if (!tipEl) return;
    const pad = 14;
    const tw = tipEl.offsetWidth;
    const th = tipEl.offsetHeight;
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    if (x + tw > window.innerWidth - 8)  x = evt.clientX - tw - pad;
    if (y + th > window.innerHeight - 8) y = evt.clientY - th - pad;
    tipEl.style.left = x + "px";
    tipEl.style.top  = y + "px";
  }

  // -------- render --------

  function buildSVG(data, width) {
    const height = width;
    const innerW = width  - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top  - MARGIN.bottom;

    const { x_min, x_max, y_min, y_max } = data.bounds;
    const xPad = (x_max - x_min) * 0.04;
    const yPad = (y_max - y_min) * 0.04;
    const xScale = v => MARGIN.left + ((v - (x_min - xPad)) / ((x_max + xPad) - (x_min - xPad))) * innerW;
    const yScale = v => MARGIN.top  + (1 - (v - (y_min - yPad)) / ((y_max + yPad) - (y_min - yPad))) * innerH;

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      width: width,
      height: height,
      style: "display: block; max-width: 100%; height: auto;",
    });

    // Cluster colour map
    const clusterIdToColor = {};
    data.clusters.forEach((c, i) => {
      clusterIdToColor[c.id] = CLUSTER_PALETTE[i % CLUSTER_PALETTE.length];
    });

    // Noise points
    const noise = data.points.filter(p => p.cluster === -1);
    if (noise.length) {
      const g = el("g", { class: "cluster-noise" });
      noise.forEach(p => {
        const cx = xScale(p.x), cy = yScale(p.y);
        if (p.is_deployment) {
          g.appendChild(el("path", {
            d: starPath(cx, cy, 5), fill: NOISE_COLOR, opacity: 0.55,
          }));
        } else {
          g.appendChild(el("circle", {
            cx, cy, r: 3, fill: NOISE_COLOR, opacity: 0.4,
          }));
        }
      });
      svg.appendChild(g);
    }

    // Per-cluster groups (so hover targets one whole cluster at a time)
    data.clusters.forEach(cluster => {
      const color = clusterIdToColor[cluster.id];
      const g = el("g", {
        class: "cluster-group",
        "data-cluster": String(cluster.id),
      });
      // Invisible larger hit box per point so hover is forgiving.
      data.points.filter(p => p.cluster === cluster.id).forEach(p => {
        const cx = xScale(p.x), cy = yScale(p.y);
        if (p.is_deployment) {
          g.appendChild(el("path", {
            d: starPath(cx, cy, 7),
            fill: color, stroke: INK, "stroke-width": 0.5, opacity: 0.85,
          }));
        } else {
          g.appendChild(el("circle", {
            cx, cy, r: 4,
            fill: color, stroke: INK, "stroke-width": 0.3, opacity: 0.7,
          }));
        }
      });
      svg.appendChild(g);
    });

    // Cluster keyword labels at each centroid
    data.clusters.forEach(cluster => {
      const color = clusterIdToColor[cluster.id];
      const cx = xScale(cluster.centroid[0]);
      const cy = yScale(cluster.centroid[1]);
      const kws = (cluster.keywords || []).slice(0, 2);
      if (!kws.length) return;
      const text = el("text", {
        x: cx, y: cy,
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        "font-family": "Tomato Grotesk, sans-serif",
        "font-size": 11,
        "font-weight": 600,
        fill: color,
        class: "cluster-label",
        "data-cluster": String(cluster.id),
        "paint-order": "stroke",
        stroke: "#FFFFFF",
        "stroke-width": 3.5,
        "stroke-linejoin": "round",
      });
      kws.forEach((k, i) => {
        text.appendChild(el("tspan", {
          x: cx,
          dy: i === 0 ? -((kws.length - 1) * 6) : 13,
        }, [`"${k}"`]));
      });
      svg.appendChild(text);
    });

    // Legend in the bottom-left
    const lg = el("g", { transform: `translate(${MARGIN.left + 4}, ${height - MARGIN.bottom + 18})` });
    lg.appendChild(el("circle", { cx: 4, cy: 4, r: 4, fill: MUTED }));
    lg.appendChild(el("text", {
      x: 16, y: 8,
      "font-family": "Univers, sans-serif", "font-size": 11, fill: INK,
    }, ["Edge ML method"]));
    lg.appendChild(el("path", { d: starPath(4, 26, 6), fill: MUTED }));
    lg.appendChild(el("text", {
      x: 16, y: 30,
      "font-family": "Univers, sans-serif", "font-size": 11, fill: INK,
    }, ["Real-world deployment"]));
    svg.appendChild(lg);

    return { svg, clusterIdToColor };
  }

  function wire(svg, data, clusterIdToColor) {
    svgRoot = svg;
    const clustersById = {};
    data.clusters.forEach(c => { clustersById[c.id] = c; });

    svg.querySelectorAll(".cluster-group").forEach(g => {
      const id = Number(g.getAttribute("data-cluster"));
      const cluster = clustersById[id];
      if (!cluster) return;
      const color = clusterIdToColor[id];
      g.style.cursor = "pointer";
      g.addEventListener("mouseenter", e => showCluster(cluster, color, e));
      g.addEventListener("mousemove",  e => positionTip(e));
      g.addEventListener("mouseleave", () => scheduleClear());
    });

    svg.querySelectorAll(".cluster-label").forEach(t => {
      const id = Number(t.getAttribute("data-cluster"));
      const cluster = clustersById[id];
      if (!cluster) return;
      const color = clusterIdToColor[id];
      t.style.cursor = "pointer";
      t.addEventListener("mouseenter", e => showCluster(cluster, color, e));
      t.addEventListener("mousemove",  e => positionTip(e));
      t.addEventListener("mouseleave", () => scheduleClear());
    });
  }

  function render(mount, data) {
    mount.innerHTML = "";
    const width = Math.max(420, Math.floor(mount.clientWidth || mount.getBoundingClientRect().width));
    const { svg, clusterIdToColor } = buildSVG(data, width);
    mount.appendChild(svg);
    wire(svg, data, clusterIdToColor);
  }

  async function boot() {
    const mount = document.getElementById(MOUNT_ID);
    if (!mount) return;
    try {
      const res = await fetch(DATA_URL, { cache: "no-cache" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      render(mount, data);

      let resizeTimer = null;
      window.addEventListener("resize", () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => render(mount, data), 120);
      });
    } catch (err) {
      console.error("fig-literature-clusters:", err);
      mount.innerHTML = '<p class="placeholder">[ figure failed to load ]</p>';
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
