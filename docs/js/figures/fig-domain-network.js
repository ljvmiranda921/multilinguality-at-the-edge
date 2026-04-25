(function () {
  const MOUNT_ID = "fig-where";
  const DETAIL_ID = "fig-where-detail";
  const DATA_URL = "assets/data/domain_method_network.json";
  const SVG_NS = "http://www.w3.org/2000/svg";

  const COLORS = {
    ink: "#35302E",
    muted: "#6b6358",
    white: "#FFFFFF",
    rule: "#e5ddcb",
  };

  function el(name, attrs, children) {
    const e = document.createElementNS(SVG_NS, name);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (children) {
      for (const c of children) {
        e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      }
    }
    return e;
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[c]);
  }

  function linearScale(d0, d1, r0, r1) {
    const dd = d1 - d0 || 1;
    return (v) => r0 + ((v - d0) / dd) * (r1 - r0);
  }

  function edgePath(x0, y0, x1, y1) {
    const mx = (x0 + x1) / 2;
    const my = (y0 + y1) / 2;
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    const curve = 0.17 * len;
    const sign = (mx * ny - my * nx) >= 0 ? 1 : -1;
    const cx = mx + nx * curve * sign;
    const cy = my + ny * curve * sign;
    return `M ${x0} ${y0} Q ${cx} ${cy} ${x1} ${y1}`;
  }

  function adjustLabelY(labels, minGap, top, bottom) {
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) {
      if (labels[i].y - labels[i - 1].y < minGap) {
        labels[i].y = labels[i - 1].y + minGap;
      }
    }
    for (let i = labels.length - 2; i >= 0; i--) {
      if (labels[i + 1].y > bottom) labels[i + 1].y = bottom;
      if (labels[i].y > labels[i + 1].y - minGap) {
        labels[i].y = labels[i + 1].y - minGap;
      }
    }
    labels.forEach((l) => {
      l.y = Math.max(top, Math.min(bottom, l.y));
    });
  }

  function labelLines(label) {
    return String(label).split("\n");
  }

  function renderDetail(detailEl, domain) {
    if (!detailEl || !domain) return;
    const samples = domain.samples || [];
    const list = samples.length
      ? `<ol class="where-detail-list">${samples.map((p) => {
          const year = p.year ? `<span class="where-detail-year"> (${p.year})</span>` : "";
          const title = `${esc(p.title)}${year}`;
          if (p.url) return `<li><a href="${esc(p.url)}" target="_blank" rel="noopener">${title}</a></li>`;
          return `<li>${title}</li>`;
        }).join("")}</ol>`
      : `<p class="placeholder">[ no representative papers available for this domain ]</p>`;
    detailEl.innerHTML =
      `<h4 class="where-detail-head">Representative deployments in ${esc(domain.id)}</h4>` +
      list;
  }

  function render(mount, detailEl, data) {
    mount.innerHTML = "";
    const width = Math.max(420, Math.floor(mount.clientWidth || mount.getBoundingClientRect().width));
    const height = width;
    const margin = 28;
    const labelPad = 34;

    const xScale = linearScale(
      data.bounds.x_min,
      data.bounds.x_max,
      margin + labelPad,
      width - margin - labelPad
    );
    const yScale = linearScale(
      data.bounds.y_min,
      data.bounds.y_max,
      height - margin - labelPad,
      margin + labelPad
    );

    const domainById = Object.fromEntries(data.domains.map((d) => [d.id, d]));
    const techById = Object.fromEntries(data.techniques.map((t) => [t.id, t]));

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      width,
      height,
      style: "display:block;max-width:100%;height:auto;",
    });

    const gEdges = el("g");
    const gLeaders = el("g");
    const gNodes = el("g");
    const gLabels = el("g");

    const labelPoints = data.techniques.map((t) => {
      const lx = xScale(t.label_x);
      const ly = yScale(t.label_y);
      return {
        id: t.id,
        x: lx,
        y: ly,
        side: (t.label_anchor === "end" ? "left" : (t.label_anchor === "start" ? "right" : (lx < width / 2 ? "left" : "right"))),
      };
    });
    const left = labelPoints.filter((p) => p.side === "left");
    const right = labelPoints.filter((p) => p.side === "right");
    adjustLabelY(left, 16, margin, height - margin);
    adjustLabelY(right, 16, margin, height - margin);
    const labelById = Object.fromEntries([...left, ...right].map((p) => [p.id, p]));

    data.edges.forEach((edge) => {
      const d = domainById[edge.domain];
      const t = techById[edge.technique];
      if (!d || !t) return;
      const x0 = xScale(d.x);
      const y0 = yScale(d.y);
      const x1 = xScale(t.x);
      const y1 = yScale(t.y);
      gEdges.appendChild(el("path", {
        d: edgePath(x0, y0, x1, y1),
        fill: "none",
        stroke: edge.color || COLORS.muted,
        "stroke-width": 0.8 + edge.weight * 0.9,
        "stroke-linecap": "round",
        opacity: 0.26,
        class: "where-edge",
        "data-domain": edge.domain,
        "data-technique": edge.technique,
      }));
    });

    data.techniques.forEach((t) => {
      const x = xScale(t.x);
      const y = yScale(t.y);
      const radius = Math.max(7.5, Math.min(16, 5 + Math.sqrt((t.size || 320) / Math.PI) * 0.5));
      gNodes.appendChild(el("circle", {
        cx: x,
        cy: y,
        r: radius,
        fill: t.color || "#B5BDC8",
        stroke: COLORS.white,
        "stroke-width": 0.9,
        class: "where-tech",
        "data-technique": t.id,
      }));

      const lp = labelById[t.id];
      if (!lp) return;
      gLeaders.appendChild(el("line", {
        x1: x,
        y1: y,
        x2: lp.x,
        y2: lp.y,
        stroke: COLORS.muted,
        "stroke-width": 0.75,
        opacity: 0.45,
        class: "where-leader",
        "data-technique": t.id,
      }));

      const anchor = lp.side === "left" ? "end" : "start";
      const text = el("text", {
        x: lp.x + (anchor === "end" ? -3 : 3),
        y: lp.y,
        "text-anchor": anchor,
        "dominant-baseline": "middle",
        "font-family": "Univers, sans-serif",
        "font-size": 9.5,
        fill: COLORS.ink,
        class: "where-label",
        "data-technique": t.id,
      });
      labelLines(t.id).forEach((line, i) => {
        text.appendChild(el("tspan", {
          x: lp.x + (anchor === "end" ? -3 : 3),
          dy: i === 0 ? 0 : 11,
        }, [line]));
      });
      gLabels.appendChild(text);
    });

    data.domains.forEach((d) => {
      const x = xScale(d.x);
      const y = yScale(d.y);
      gNodes.appendChild(el("circle", {
        cx: x,
        cy: y,
        r: 34,
        fill: d.color || COLORS.muted,
        stroke: COLORS.ink,
        "stroke-width": 1.2,
        class: "where-domain",
        "data-domain": d.id,
      }));

      const txt = el("text", {
        x,
        y,
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        "font-family": "Tomato Grotesk, sans-serif",
        "font-size": 11.5,
        "font-weight": 700,
        fill: COLORS.white,
        class: "where-domain",
        "data-domain": d.id,
      });
      const lines = labelLines(d.label || d.id);
      lines.forEach((line, i) => {
        txt.appendChild(el("tspan", {
          x,
          dy: i === 0 ? (lines.length > 1 ? -4.5 : 0) : 11,
        }, [line]));
      });
      gNodes.appendChild(txt);
    });

    svg.appendChild(gEdges);
    svg.appendChild(gLeaders);
    svg.appendChild(gNodes);
    svg.appendChild(gLabels);
    mount.appendChild(svg);

    const techForDomain = {};
    data.edges.forEach((e) => {
      if (!techForDomain[e.domain]) techForDomain[e.domain] = new Set();
      techForDomain[e.domain].add(e.technique);
    });

    function setActive(domainId) {
      svg.classList.add("has-active");
      svg.querySelectorAll(".is-active").forEach((n) => n.classList.remove("is-active"));

      svg.querySelectorAll(`.where-domain[data-domain="${CSS.escape(domainId)}"]`)
        .forEach((n) => n.classList.add("is-active"));

      const activeTech = techForDomain[domainId] || new Set();
      svg.querySelectorAll(`.where-edge[data-domain="${CSS.escape(domainId)}"]`)
        .forEach((n) => n.classList.add("is-active"));

      activeTech.forEach((techId) => {
        svg.querySelectorAll(`.where-tech[data-technique="${CSS.escape(techId)}"], .where-label[data-technique="${CSS.escape(techId)}"], .where-leader[data-technique="${CSS.escape(techId)}"]`)
          .forEach((n) => n.classList.add("is-active"));
      });

      renderDetail(detailEl, domainById[domainId]);
    }

    svg.querySelectorAll(".where-domain[data-domain]").forEach((node) => {
      node.style.cursor = "pointer";
      node.addEventListener("click", () => setActive(node.getAttribute("data-domain")));
    });

    if (data.domains.length) setActive(data.domains[0].id);
  }

  async function boot() {
    const mount = document.getElementById(MOUNT_ID);
    if (!mount) return;
    const detail = document.getElementById(DETAIL_ID);
    try {
      const res = await fetch(DATA_URL, { cache: "no-cache" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      render(mount, detail, data);

      let timer = null;
      window.addEventListener("resize", () => {
        clearTimeout(timer);
        timer = setTimeout(() => render(mount, detail, data), 120);
      });
    } catch (err) {
      console.error("fig-domain-network:", err);
      mount.innerHTML = '<p class="placeholder">[ figure failed to load ]</p>';
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
