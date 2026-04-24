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

const COLS = 140;
const ROWS = 48;
const STAGE_XS = [-0.8, -0.4, 0.0, 0.4, 0.8];
const PARTICLES_PER_STAGE_PER_SIDE = 4;

// Each particle is assigned to a stage and stays within that stage's
// column for its entire lifetime. The particle's x position at any
// height is (stage_x + aNormOffset * columnWidth(y)), where
// columnWidth(y) is the same hourglass profile used by the mesh. So
// as the column narrows toward the waist, the particle narrows with
// it and never crosses the whitespace into a neighbouring column.
function buildParticlesGeometry() {
  const total = STAGE_XS.length * PARTICLES_PER_STAGE_PER_SIDE * 2;
  // three.js requires a position attribute — unused here, we compute
  // everything from the other attributes in the vertex shader.
  const position     = new Float32Array(total * 3);
  const aStageX      = new Float32Array(total);
  const aNormOffset  = new Float32Array(total); // [-0.85, 0.85] within column
  const aStartY      = new Float32Array(total);
  const aSpawnOffset = new Float32Array(total);
  const aSpeed       = new Float32Array(total);

  let i = 0;
  for (let s = 0; s < STAGE_XS.length; s++) {
    const sx = STAGE_XS[s];
    for (let side = 0; side < 2; side++) {
      const dir = side === 0 ? 1 : -1; // +1 top, -1 bottom
      for (let k = 0; k < PARTICLES_PER_STAGE_PER_SIDE; k++) {
        aStageX[i]      = sx;
        aNormOffset[i]  = (Math.random() - 0.5) * 1.7;  // ≈ [-0.85, 0.85]
        aStartY[i]      = dir * (0.82 + Math.random() * 0.16);
        aSpawnOffset[i] = Math.random() * 3.0;
        aSpeed[i]       = 0.28 + Math.random() * 0.22;
        i++;
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position',     new THREE.BufferAttribute(position, 3));
  geo.setAttribute('aStageX',      new THREE.BufferAttribute(aStageX, 1));
  geo.setAttribute('aNormOffset',  new THREE.BufferAttribute(aNormOffset, 1));
  geo.setAttribute('aStartY',      new THREE.BufferAttribute(aStartY, 1));
  geo.setAttribute('aSpawnOffset', new THREE.BufferAttribute(aSpawnOffset, 1));
  geo.setAttribute('aSpeed',       new THREE.BufferAttribute(aSpeed, 1));
  return geo;
}

const PARTICLE_VERT = `
  attribute float aStageX;
  attribute float aNormOffset;
  attribute float aStartY;
  attribute float aSpawnOffset;
  attribute float aSpeed;
  uniform float   uTime;
  uniform float   uPointSize;
  uniform float   uNarrow;      // must match the mesh material
  uniform float   uWide;
  varying float   vAlpha;
  varying float   vTop;

  void main() {
    // Normalised lifetime progress in [0, 1], looping.
    float t = fract((uTime + aSpawnOffset) * aSpeed);

    // y descends from |aStartY| toward 0 along the same easing used
    // by the mesh pinch.
    float curY = mix(aStartY, 0.0, smoothstep(0.0, 1.0, t));

    // Hourglass width at the particle's current height — identical
    // profile to the mesh so particles stay inside the visible grid.
    float yF    = smoothstep(0.0, 1.0, abs(curY));
    float width = mix(uNarrow, uWide, yF);

    // x is offset from the stage centre by a fixed fraction of the
    // column width, so the particle follows the column's taper.
    float curX = aStageX + aNormOffset * width;

    // Fade in at spawn, fade out just before the waist.
    float fadeIn  = smoothstep(0.0, 0.12, t);
    float fadeOut = smoothstep(1.0, 0.78, t);
    vAlpha = fadeIn * fadeOut;
    vTop   = aStartY > 0.0 ? 1.0 : 0.0;

    gl_Position  = projectionMatrix * modelViewMatrix * vec4(curX, curY, 0.0, 1.0);
    gl_PointSize = uPointSize;
  }
`;

const PARTICLE_FRAG = `
  uniform vec3  uAccentCool;   // blue  (from top, Edge)
  uniform vec3  uAccentWarm;   // terracotta (from bottom, Multilinguality)
  uniform float uOpacity;
  varying float vAlpha;
  varying float vTop;

  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv);
    if (d > 0.5) discard;
    // Soft disc with a slight core brightness.
    float shape = smoothstep(0.5, 0.12, d);
    vec3  col   = mix(uAccentWarm, uAccentCool, vTop);
    gl_FragColor = vec4(col, vAlpha * shape * uOpacity);
  }
`;

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
    // Smoothstep gives an S-curve hourglass: the waist glides into
    // the wide ends instead of kinking.
    float yFactor = smoothstep(0.0, 1.0, abs(pos.y));
    float width = mix(uNarrow, uWide, yFactor);
    width *= 1.0 + 0.05 * sin(t * 0.35);

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
    float pinchT = smoothstep(0.0, 1.0, pipeN);
    pos.x = mix(pos.x, nearStage, pinchT * 0.7);

    // ---- 3. Subtle drift so the mesh is never fully static ----
    pos.y += sin(pos.x *  5.5 + t * 0.45) * 0.010 * mass;
    pos.x += cos(pos.y *  4.0 + t * 0.3 ) * 0.004 * mass;

    vMass = mass;
    vMid  = 1.0 - abs(pos.y);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const FRAG = `
  uniform vec3  uInk;
  uniform vec3  uAccentCool;   // Edge — blue, top half
  uniform vec3  uAccentWarm;   // Multilinguality — terracotta, bottom half
  uniform float uOpacity;
  uniform float uTime;
  varying float vMass;
  varying float vMid;
  varying float vY;

  void main() {
    // Top/bottom tint selection — top half skews blue, bottom half
    // skews terracotta, blended smoothly across the waist.
    float topWeight = smoothstep(-0.15, 0.15, vY);
    // Subtle ambient tint throughout (0.22) that strengthens at the
    // waist (up to ~0.85) for the collision emphasis.
    float midEmph = smoothstep(0.45, 1.0, vMid);
    float tintAmt = 0.22 + midEmph * 0.65;

    vec3 topCol = mix(uInk, uAccentCool, tintAmt);
    vec3 botCol = mix(uInk, uAccentWarm, tintAmt);
    vec3 col    = mix(botCol, topCol, topWeight);

    // Flow bands: soft crests travel FROM the edges TOWARD the
    // middle. Wider wavelength + slower speed than before, so the
    // motion reads as breathing rather than strobing.
    float flowPhase = abs(vY) * 4.5 + uTime * 1.2;
    float flow      = 0.55 + 0.45 * sin(flowPhase);

    // Base boost: the top and bottom edges of the mesh (where it
    // meets the requirement bands) get extra opacity so the "force
    // origin" reads clearly.
    float baseBoost = smoothstep(0.55, 1.0, abs(vY));

    float waistPulse = 0.78 + 0.22 * sin(uTime * 1.1);
    float a = uOpacity
            * vMass
            * (0.35 + flow * 0.7)
            * (0.7 + midEmph * waistPulse * 0.55);
    a *= 1.0 + baseBoost * 0.9;

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
      uTime:        { value: 0 },
      uInk:         { value: new THREE.Color(0x35302e) }, // --ink
      uAccentCool:  { value: new THREE.Color(0x254eff) }, // blue (Edge)
      uAccentWarm:  { value: new THREE.Color(0xc96a2e) }, // terracotta (Multilinguality)
      uOpacity:     { value: 0.55 },
      uNarrow:      { value: 0.03 }, // waist half-width
      uWide:        { value: 0.16 }, // edge half-width
    },
    vertexShader:   VERT,
    fragmentShader: FRAG,
    transparent:    true,
  });

  const mesh = new THREE.LineSegments(geometry, material);
  scene.add(mesh);

  // Particle layer — shares the uTime uniform with the mesh so
  // everything stays on the same clock.
  const particleMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uTime:       material.uniforms.uTime,
      uAccentCool: material.uniforms.uAccentCool,
      uAccentWarm: material.uniforms.uAccentWarm,
      uNarrow:     material.uniforms.uNarrow, // share with mesh
      uWide:       material.uniforms.uWide,
      uOpacity:    { value: 0.9 },
      uPointSize:  { value: 4.0 },
    },
    vertexShader:   PARTICLE_VERT,
    fragmentShader: PARTICLE_FRAG,
    transparent:    true,
    depthTest:      false,
  });
  const particles = new THREE.Points(buildParticlesGeometry(), particleMaterial);
  scene.add(particles);

  function resize() {
    const { width, height } = container.getBoundingClientRect();
    if (width === 0 || height === 0) return;
    renderer.setSize(width, height, false);
    // Scale point size with DPR + overall width so particles stay
    // a consistent visual size across devices.
    const dpr = renderer.getPixelRatio();
    particleMaterial.uniforms.uPointSize.value = Math.min(6.0, Math.max(3.0, width / 240)) * dpr;
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
