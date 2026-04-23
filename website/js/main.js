document.addEventListener("DOMContentLoaded", () => {
  const pipelineMount = document.getElementById("pipeline-figure");
  if (pipelineMount && window.PipelineModule) {
    window.PipelineModule.renderPipeline(pipelineMount);
    window.PipelineModule.resetDetail();
  }

  // Chart mounts (#fig-how, #fig-who, #fig-where) stay as placeholders
  // until wired to Plotly/cytoscape in a follow-up.
});
