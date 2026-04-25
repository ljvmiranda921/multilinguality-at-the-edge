(function () {
  const MOUNT_ID = "fig-1";
  const DATA_URL = "assets/data/infra_lingdiv_ict.json";

  const GROUP_ORDER = [
    "Low-income",
    "Lower-middle-income",
    "Upper-middle-income",
    "High-income",
  ];

  const GROUP_STYLE = {
    "Low-income":          { color: "#C96A2E", symbol: "circle" },
    "Lower-middle-income": { color: "#E39A5F", symbol: "square" },
    "Upper-middle-income": { color: "#7B93B8", symbol: "diamond" },
    "High-income":         { color: "#254EFF", symbol: "triangle-up" },
  };

  const INK        = "#35302E";
  const RULE       = "#e5ddcb";
  const FONT_BODY  = "Univers, Helvetica, Arial, sans-serif";
  const FONT_HEAD  = "'Tomato Grotesk', Helvetica, Arial, sans-serif";

  function buildTraces(rows) {
    return GROUP_ORDER.map((group) => {
      const pts = rows.filter((r) => r.income_group === group);
      const style = GROUP_STYLE[group];
      return {
        type: "scatter",
        mode: "markers",
        name: group,
        x: pts.map((r) => r.internet_users),
        y: pts.map((r) => r.num_living_languages),
        text: pts.map((r) => r.country),
        customdata: pts.map((r) => r.income_group),
        hovertemplate:
          "<b>%{text}</b><br>" +
          "Internet users: %{x:.1f}%<br>" +
          "Living languages: %{y}<br>" +
          "%{customdata}<extra></extra>",
        marker: {
          color: style.color,
          symbol: style.symbol,
          size: 10,
          opacity: 0.85,
          line: { color: INK, width: 0.6 },
        },
      };
    });
  }

  function buildAnnotations(rows) {
    return rows
      .filter((r) => r.annotate)
      .map((r) => {
        const color = GROUP_STYLE[r.income_group]?.color || INK;
        // Iceland is dense in the upper right; shift its label left.
        const iceland = r.country === "Iceland";
        return {
          x: r.internet_users,
          y: Math.log10(r.num_living_languages),
          yref: "y",
          xref: "x",
          text: r.label,
          showarrow: false,
          xanchor: iceland ? "right" : "left",
          xshift: iceland ? -8 : 7,
          yshift: 4,
          font: { family: FONT_BODY, size: 11, color },
        };
      });
  }

  function render(mount, rows) {
    mount.innerHTML = "";
    const chartDiv = document.createElement("div");
    chartDiv.style.width = "100%";
    chartDiv.style.height = "100%";
    chartDiv.style.flex = "1 1 auto";
    chartDiv.style.alignSelf = "stretch";
    mount.appendChild(chartDiv);
    mount.style.padding = "0.5rem";
    mount.style.alignItems = "stretch";
    mount.style.justifyContent = "stretch";

    const traces = buildTraces(rows);
    const annotations = buildAnnotations(rows);

    const layout = {
      margin: { l: 56, r: 14, t: 10, b: 80 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor:  "rgba(0,0,0,0)",
      font:          { family: FONT_BODY, color: INK, size: 12 },
      xaxis: {
        title: {
          text: "% Individuals using the Internet (2023)",
          font: { family: FONT_HEAD, size: 13 },
          standoff: 12,
        },
        range: [-4, 104],
        gridcolor: RULE,
        zerolinecolor: RULE,
        linecolor: INK,
        showline: true,
        ticks: "outside",
        tickcolor: INK,
        tickfont: { family: FONT_BODY, size: 11 },
      },
      yaxis: {
        title: {
          text: "Number of living languages",
          font: { family: FONT_HEAD, size: 13 },
          standoff: 8,
        },
        type: "log",
        gridcolor: RULE,
        zerolinecolor: RULE,
        linecolor: INK,
        showline: true,
        ticks: "outside",
        tickcolor: INK,
        tickfont: { family: FONT_BODY, size: 11 },
      },
      legend: {
        orientation: "h",
        y: -0.22,
        x: 0.5,
        xanchor: "center",
        font: { family: FONT_BODY, size: 11 },
        bgcolor: "rgba(0,0,0,0)",
      },
      hoverlabel: {
        bgcolor: "#ffffff",
        bordercolor: INK,
        font: { family: FONT_BODY, size: 12, color: INK },
      },
      annotations,
    };

    const config = {
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: [
        "lasso2d", "select2d", "autoScale2d",
        "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian",
      ],
      toImageButtonOptions: { filename: "infra_lingdiv_ict", format: "svg" },
    };

    window.Plotly.newPlot(chartDiv, traces, layout, config);
  }

  async function boot() {
    const mount = document.getElementById(MOUNT_ID);
    if (!mount) return;
    if (!window.Plotly) {
      // Plotly CDN hasn't finished loading yet — retry briefly.
      return void setTimeout(boot, 50);
    }
    try {
      const res = await fetch(DATA_URL, { cache: "no-cache" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      render(mount, json.data);
    } catch (err) {
      console.error("fig-infra-lingdiv:", err);
      mount.innerHTML =
        '<p class="placeholder">[ figure failed to load ]</p>';
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
