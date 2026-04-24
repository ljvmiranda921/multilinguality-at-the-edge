// Animated wireframe mesh rendered behind the pipeline strip.
//
// The mesh is not a full rectangle — it's shaped as five stalactite/
// stalagmite columns, one per pipeline stage. Each column is wide
// at the top and bottom edges and narrows to a tight waist at the
// pipeline (y = 0), with clear whitespace between columns. The
// colour shifts from ink at the edges to blue at the waist, and a
// soft pulse suggests the ongoing interaction between the edge and
// multilingual requirements colliding at each stage.
//
// No build step — three.js is imported as an ES module from a CDN.

import * as THREE from 'https://esm.sh/three@0.168.0';

const COLS = 80;
const ROWS = 32;

function buildGridGeometry() {
  const positions = [];
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const px = (x / (COLS - 1)) * 2 - 1;   // [-1, 1]
      const py = (y / (ROWS - 1)) * 2 - 1;   // [-1, 1]
      positions.push(px, py, 0);
    }
  }

  const indices = [];
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS - 1; x++) {
      indices.push(y * COLS + x, y * COLS + x + 1);
    }
  }
  for (let y = 0; y < ROWS - 1; y++) {
    for (let x = 0; x < COLS; x++) {
      indices.push(y * COLS + x, (y + 1) * COLS + x);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);
  return geo;
}

const VERT = `
  uniform float uTime;
  uniform float uNarrow;   // column half-width at the pipeline waist
  uniform float uWide;     // column half-width at the top/bottom edges
  varying float vMass;     // 0 = whitespace, 1 = inside a flow column
  varying float vMid;      // 1 = at pipeline centre, 0 = at edges
  varying float vY;        // original y, for flow-band animation

  void main() {
    vec3 pos = position;
    float t = uTime;
    vY = pos.y;  // keep the untransformed y for the fragment shader

    // Five equal-spaced stage columns, matching the .pipeline-segment
    // positions on the strip.
    float stageXs[5];
    stageXs[0] = -0.8;
    stageXs[1] = -0.4;
    stageXs[2] =  0.0;
    stageXs[3] =  0.4;
    stageXs[4] =  0.8;

    // ---- 1. Mass (where the column renders) ----
    // Width is narrow at the waist (y = 0) and wide at the edges.
    // Power > 1 steepens the taper so the pinch reads clearly.
    float yFactor = pow(abs(pos.y), 1.15);
    float width = mix(uNarrow, uWide, yFactor);
    width *= 1.0 + 0.06 * sin(t * 0.4);

    float mass = 0.0;
    float nearStage = stageXs[0];
    float minD = 10.0;
    for (int i = 0; i < 5; i++) {
      float dx = pos.x - stageXs[i];
      mass = max(mass, exp(-dx * dx / (width * width)));
      float d = abs(dx);
      if (d < minD) { minD = d; nearStage = stageXs[i]; }
    }

    // ---- 2. Pinch (lines curve inward toward the stage centre
    //         as they approach the pipeline) ----
    float pipeN  = 1.0 - abs(pos.y);
    float pinchT = pow(pipeN, 1.3);
    pos.x = mix(pos.x, nearStage, pinchT * 0.65);

    // ---- 3. Subtle drift so the mesh is never fully static ----
    pos.y += sin(pos.x * 14.0 + t * 0.8) * 0.006 * mass;

    vMass = mass;
    vMid  = 1.0 - abs(pos.y);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const FRAG = `
  uniform vec3  uInk;
  uniform vec3  uAccent;
  uniform float uOpacity;
  uniform float uTime;
  varying float vMass;
  varying float vMid;
  varying float vY;

  void main() {
    // Colour: ink at the wide ends, blue at the waist. Smoothstep
    // sharpens the transition so the blue reads as a distinct
    // collision band rather than a soft gradient.
    float midEmph = smoothstep(0.45, 1.0, vMid);
    vec3  col     = mix(uInk, uAccent, midEmph * 0.85);

    // Flow bands: bright crests travel FROM the edges TOWARD the
    // middle, selling "force flowing into the pipeline stages."
    // |vY| grows toward the edges; adding uTime shifts the pattern
    // so a crest is at decreasing |vY| over time.
    float flowPhase = abs(vY) * 7.0 + uTime * 1.6;
    float flow      = 0.5 + 0.5 * sin(flowPhase);

    float waistPulse = 0.78 + 0.22 * sin(uTime * 1.1);
    float a = uOpacity
            * vMass
            * (0.35 + flow * 0.7)
            * (0.7 + midEmph * waistPulse * 0.55);

    gl_FragColor = vec4(col, a);
  }
`;

function mountPipelineMesh(container) {
  if (!container) return;

  if (window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.className = 'pipeline-mesh-canvas';
  canvas.setAttribute('aria-hidden', 'true');
  container.insertBefore(canvas, container.firstChild);

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
  } catch (e) {
    canvas.remove();
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);

  const scene  = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.z = 1;

  const geometry = buildGridGeometry();
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime:    { value: 0 },
      uInk:     { value: new THREE.Color(0x35302e) }, // --ink
      uAccent:  { value: new THREE.Color(0x254eff) }, // --blue
      uOpacity: { value: 0.55 },
      uNarrow:  { value: 0.03 }, // waist half-width (pipeline centre)
      uWide:    { value: 0.16 }, // edge half-width (top + bottom)
    },
    vertexShader:   VERT,
    fragmentShader: FRAG,
    transparent:    true,
  });

  const mesh = new THREE.LineSegments(geometry, material);
  scene.add(mesh);

  function resize() {
    const { width, height } = container.getBoundingClientRect();
    if (width === 0 || height === 0) return;
    renderer.setSize(width, height, false);
  }
  resize();

  const ro = new ResizeObserver(resize);
  ro.observe(container);

  const clock = new THREE.Clock();
  let running = !document.hidden;

  document.addEventListener('visibilitychange', () => {
    running = !document.hidden;
    if (running) clock.start();
  });

  function frame() {
    requestAnimationFrame(frame);
    if (!running) return;
    material.uniforms.uTime.value = clock.getElapsedTime();
    renderer.render(scene, camera);
  }
  frame();
}

document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('pipeline-figure');
  mountPipelineMesh(container);
});
