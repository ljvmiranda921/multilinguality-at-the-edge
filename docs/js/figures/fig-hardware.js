// Four slowly spinning hardware classes, built procedurally (units: mm).

import * as THREE from 'https://esm.sh/three@0.168.0';

// ---------- materials ----------

const M = {
  pcbBlack: new THREE.MeshStandardMaterial({ color: 0x1c1e22, roughness: 0.55 }),
  pcbGreen: new THREE.MeshStandardMaterial({ color: 0x1f6f3c, roughness: 0.5 }),
  pcbBlue:  new THREE.MeshStandardMaterial({ color: 0x1f2f4d, roughness: 0.5 }),
  metal:    new THREE.MeshStandardMaterial({ color: 0xd2d5da, metalness: 1, roughness: 0.28 }),
  brushed:  new THREE.MeshStandardMaterial({ color: 0xb9bdc4, metalness: 0.85, roughness: 0.42 }),
  gold:     new THREE.MeshStandardMaterial({ color: 0xd9ab48, metalness: 1, roughness: 0.32 }),
  plastic:  new THREE.MeshStandardMaterial({ color: 0x121314, roughness: 0.5 }),
  chip:     new THREE.MeshStandardMaterial({ color: 0x18191b, roughness: 0.35 }),
  usb3:     new THREE.MeshStandardMaterial({ color: 0x2457c5, roughness: 0.5 }),
  white:    new THREE.MeshStandardMaterial({ color: 0xeeeae2, roughness: 0.6 }),
  glass:    new THREE.MeshStandardMaterial({ color: 0x050607, metalness: 0.2, roughness: 0.08 }),
  phone:    new THREE.MeshStandardMaterial({ color: 0x2b4a7a, metalness: 0.25, roughness: 0.3 }),
  keys:     new THREE.MeshStandardMaterial({ color: 0x2a2c31, roughness: 0.65 }),
  well:     new THREE.MeshStandardMaterial({ color: 0x8e939b, metalness: 0.6, roughness: 0.5 }),
  ledRed:   new THREE.MeshStandardMaterial({ color: 0xff3b30, emissive: 0xff2a1f, emissiveIntensity: 1.5 }),
  ledGreen: new THREE.MeshStandardMaterial({ color: 0x35d06b, emissive: 0x22c55e, emissiveIntensity: 1.5 }),
};

// ---------- helpers ----------

function box(w, h, d, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  return m;
}

function cyl(r, h, mat, x = 0, y = 0, z = 0, seg = 32) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, seg), mat);
  m.position.set(x, y, z);
  return m;
}

function roundedRectShape(w, h, r) {
  const s = new THREE.Shape();
  const x = -w / 2, y = -h / 2;
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y);
  s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r);
  s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h);
  s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r);
  s.quadraticCurveTo(x, y, x + r, y);
  return s;
}

// Rounded slab in the XY plane, centred, thickness along Z.
function slab(w, h, t, r, mat, bevel = Math.min(1.2, t / 3)) {
  const geo = new THREE.ExtrudeGeometry(roundedRectShape(w - 2 * bevel, h - 2 * bevel, Math.max(r - bevel, 0.1)), {
    depth: t - 2 * bevel, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 4, curveSegments: 10,
  });
  geo.translate(0, 0, -(t - 2 * bevel) / 2);
  return new THREE.Mesh(geo, mat);
}

function plate(w, h, r, mat) {
  const geo = new THREE.ShapeGeometry(roundedRectShape(w, h, r), 10);
  const uv = geo.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) / w + 0.5, uv.getY(i) / h + 0.5);
  return new THREE.Mesh(geo, mat);
}

function canvasTexture(w, h, draw) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

// Flat text decal lying on a horizontal surface.
function decal(text, w, h, color = '#f2f2f2', font = '600 64px Helvetica, Arial, sans-serif') {
  const tex = canvasTexture(512, Math.round(512 * h / w), (g, cw, ch) => {
    g.fillStyle = color; g.font = font; g.textBaseline = 'middle';
    g.fillText(text, 6, ch / 2);
  });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
    new THREE.MeshStandardMaterial({ map: tex, transparent: true, roughness: 0.6 }));
  m.rotation.x = -Math.PI / 2;
  return m;
}

function pinRow(count, pitch, mat, geo, positionAt) {
  const inst = new THREE.InstancedMesh(geo, mat, count);
  const m4 = new THREE.Matrix4();
  for (let i = 0; i < count; i++) {
    const [x, y, z] = positionAt(i, (i - (count - 1) / 2) * pitch);
    inst.setMatrixAt(i, m4.makeTranslation(x, y, z));
  }
  return inst;
}

function wrapText(g, text, x, y, maxW, lh) {
  let line = '';
  for (const word of text.split(' ')) {
    const test = line ? line + ' ' + word : word;
    if (g.measureText(test).width > maxW && line) { g.fillText(line, x, y); line = word; y += lh; }
    else line = test;
  }
  g.fillText(line, x, y);
  return y + lh;
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath(); g.roundRect(x, y, w, h, r); g.fill();
}

// ---------- ESP32 DevKit ----------

function buildESP32() {
  const g = new THREE.Group();
  const T = 1.6;
  g.add(box(52, T, 28, M.pcbBlack, 0, T / 2, 0));

  // WROOM module: blue carrier, metal shield, PCB antenna
  g.add(box(25.5, 0.8, 18, M.pcbBlue, -13.25, T + 0.4, 0));
  const shield = box(17.5, 2.6, 16, M.metal, -9.5, T + 0.8 + 1.3, 0);
  g.add(shield);
  const label = decal('ESP32-WROOM-32', 14, 3, '#5b6068', '600 54px Helvetica, Arial, sans-serif');
  label.position.set(-9.5, T + 0.8 + 2.61, -1.5);
  g.add(label);
  for (let i = 0; i < 5; i++) {
    g.add(box(0.5, 0.05, 13, M.gold, -24.5 + i * 1.1, T + 0.83, 0));
  }
  g.add(box(5, 0.05, 0.5, M.gold, -22.3, T + 0.83, 6.5));

  // headers underneath (pins point down), solder pads on top
  const pinGeo = new THREE.BoxGeometry(0.64, 8.5, 0.64);
  const padGeo = new THREE.CylinderGeometry(0.85, 0.85, 0.06, 16);
  for (const z of [-12.7, 12.7]) {
    g.add(box(48.3, 2.5, 2.5, M.plastic, 2, -1.25, z));
    g.add(pinRow(19, 2.54, M.gold, pinGeo, (i, o) => [2 + o, -3.5, z]));
    g.add(pinRow(19, 2.54, M.gold, padGeo, (i, o) => [2 + o, T + 0.03, z]));
  }

  // micro USB, buttons, USB-UART bridge, regulator, LED
  g.add(box(6, 2.8, 7.5, M.metal, 24, T + 1.4, 0));
  for (const z of [-8.5, 8.5]) {
    g.add(box(4, 1.4, 3, M.metal, 19.5, T + 0.7, z));
    g.add(cyl(0.9, 0.8, M.plastic, 19.5, T + 1.8, z, 16));
  }
  g.add(box(5, 1, 5, M.chip, 11, T + 0.5, 3));
  g.add(box(6.5, 1.3, 3.5, M.chip, 5, T + 0.65, -6));
  g.add(box(1.6, 0.6, 0.8, M.ledRed, 15, T + 0.3, -6));
  for (let i = 0; i < 6; i++) g.add(box(1.6, 0.5, 0.8, M.white, 2 + i * 2.2, T + 0.25, 7.5));
  return g;
}

// ---------- Raspberry Pi 5 ----------

function buildPi5() {
  const g = new THREE.Group();
  const W = 85, D = 56, T = 1.4;
  const X = (x) => x - W / 2;          // board-corner coordinates -> centred
  const Z = (z) => D / 2 - z;          // z measured from the HDMI edge

  const pcb = slab(W, D, T, 3, M.pcbGreen, 0.3);
  pcb.rotation.x = -Math.PI / 2;
  pcb.position.y = T / 2;
  g.add(pcb);
  const top = T;

  // mounting holes
  for (const [hx, hz] of [[3.5, 3.5], [61.5, 3.5], [3.5, 52.5], [61.5, 52.5]]) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(2.2, 0.6, 8, 24), M.gold);
    ring.rotation.x = Math.PI / 2;
    ring.position.set(X(hx), top + 0.02, Z(hz));
    g.add(ring);
    g.add(cyl(1.4, T + 0.1, M.plastic, X(hx), T / 2, Z(hz), 20));
  }

  // 2x20 GPIO header
  g.add(box(51, 2.5, 5, M.plastic, X(32.5), top + 1.25, Z(52.5)));
  const pinGeo = new THREE.BoxGeometry(0.64, 6, 0.64);
  for (const dz of [-1.27, 1.27]) {
    g.add(pinRow(20, 2.54, M.gold, pinGeo, (i, o) => [X(32.5) + o, top + 3.5, Z(52.5) + dz]));
  }

  // SoC with heat spreader, RAM, RP1 southbridge, PMIC
  g.add(box(17, 1.4, 17, M.metal, X(29), top + 0.7, Z(30)));
  g.add(box(12, 1, 15, M.chip, X(12), top + 0.5, Z(30)));
  g.add(box(10, 0.9, 10, M.chip, X(55), top + 0.45, Z(26)));
  g.add(box(6, 0.8, 6, M.chip, X(15), top + 0.4, Z(13)));
  const name = decal('Raspberry Pi 5', 22, 3.6, '#f4f4ef');
  name.position.set(X(40), top + 0.02, Z(42));
  g.add(name);

  // USB stacks (one USB 3 with blue tongues) and Ethernet on the right edge
  const usbFace = (mat, z) => {
    const s = new THREE.Group();
    s.add(box(17.5, 16, 13.2, M.metal, 0, 8, 0));
    for (const y of [4.2, 11.8]) {
      s.add(box(0.4, 5.2, 12, M.plastic, 8.8, y, 0));
      s.add(box(0.5, 1.7, 10, mat, 8.9, y, 0));
    }
    s.position.set(X(85 - 17.5 / 2 + 2.5), top, Z(z));
    return s;
  };
  g.add(usbFace(M.plastic, 47));
  g.add(usbFace(M.usb3, 29));
  const eth = new THREE.Group();
  eth.add(box(21, 13.5, 16, M.metal, 0, 6.75, 0));
  eth.add(box(0.4, 9, 11.5, M.plastic, 10.6, 5.6, 0));
  eth.add(box(0.4, 1.4, 2, M.ledGreen, 10.7, 11.6, -5.5));
  eth.add(box(0.4, 1.4, 2, M.ledRed, 10.7, 11.6, 5.5));
  eth.position.set(X(85 - 21 / 2 + 2.5), top, Z(10.25));
  g.add(eth);

  // USB-C power and two micro HDMI on the bottom edge
  g.add(box(9, 3.2, 7.5, M.metal, X(11.2), top + 1.6, Z(2.5)));
  for (const x of [25.8, 39.2]) g.add(box(7, 3.4, 7.5, M.metal, X(x), top + 1.7, Z(2.5)));

  // camera/display FPC connectors, PCIe FPC, fan header, power button
  for (const x of [47, 53]) g.add(box(3.5, 2.6, 16, M.plastic, X(x), top + 1.3, Z(12)));
  g.add(box(3, 2.2, 17, M.white, X(2.5), top + 1.1, Z(30)));
  g.add(box(6, 4, 3, M.white, X(72), top + 2, Z(46)));
  g.add(box(3, 1.6, 3, M.plastic, X(4), top + 0.8, Z(46)));
  return g;
}

// ---------- low-cost smartphone ----------

function phoneScreenTexture() {
  return canvasTexture(560, 1216, (g, w, h) => {
    g.fillStyle = '#ffffff'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#1a1a1a'; g.font = '500 26px Helvetica, Arial, sans-serif';
    g.fillText('09:41', 40, 54);
    g.textAlign = 'right'; g.fillText('●●● 2G', w - 40, 54); g.textAlign = 'left';
    g.beginPath(); g.arc(w / 2, 46, 14, 0, Math.PI * 2); g.fillStyle = '#050505'; g.fill();

    g.fillStyle = '#254eff'; g.fillRect(0, 90, w, 120);
    g.fillStyle = '#ffffff'; g.font = '600 36px Helvetica, Arial, sans-serif';
    g.fillText('Mshauri wa Kilimo', 40, 145);
    g.font = '300 24px Helvetica, Arial, sans-serif';
    g.fillText('on-device · offline', 40, 184);

    g.font = '300 30px Helvetica, Arial, sans-serif';
    g.fillStyle = '#f8ecda'; roundRect(g, 150, 260, 370, 140, 26);
    g.fillStyle = '#1a1a1a'; wrapText(g, 'Mahindi yangu yana madoa ya njano.', 176, 312, 320, 40);

    g.fillStyle = '#eef1ff'; roundRect(g, 40, 440, 400, 220, 26);
    g.fillStyle = '#1a1a1a'; wrapText(g, 'Huenda ni upungufu wa naitrojeni. Jaribu kuongeza samadi.', 66, 492, 350, 40);

    g.fillStyle = '#f8ecda'; roundRect(g, 260, 700, 260, 90, 26);
    g.fillStyle = '#1a1a1a'; g.fillText('Asante!', 290, 756);

    g.fillStyle = '#f2f2f2'; roundRect(g, 30, h - 150, w - 60, 86, 43);
    g.fillStyle = '#8a8a8a'; g.fillText('Andika ujumbe…', 66, h - 96);
    g.fillStyle = '#254eff'; g.beginPath(); g.arc(w - 82, h - 107, 30, 0, Math.PI * 2); g.fill();
  });
}

function buildPhone() {
  const g = new THREE.Group();
  const W = 76, H = 164, T = 8.6;
  g.add(slab(W, H, T, 10, M.phone, 1.6));

  const front = plate(W - 2, H - 2, 9, M.glass);
  front.position.z = T / 2 + 0.01;
  g.add(front);
  const screen = plate(W - 6, H - 8, 7, new THREE.MeshBasicMaterial({ map: phoneScreenTexture(), toneMapped: false }));
  screen.position.set(0, -1, T / 2 + 0.03);
  g.add(screen);

  // rear camera island, lenses, flash
  const island = slab(26, 30, 1.6, 6, M.glass, 0.5);
  island.position.set(-18, 55, -T / 2 - 0.6);
  g.add(island);
  for (const y of [63, 50]) {
    const ring = cyl(5.2, 1.4, M.metal, -18, y, -T / 2 - 1.2);
    ring.rotation.x = Math.PI / 2;
    const lens = cyl(4.2, 1.5, M.glass, -18, y, -T / 2 - 1.25);
    lens.rotation.x = Math.PI / 2;
    g.add(ring, lens);
  }
  const flash = cyl(1.6, 0.5, M.white, -6, 63, -T / 2 - 0.3, 16);
  flash.rotation.x = Math.PI / 2;
  g.add(flash);

  // side buttons
  g.add(box(1.2, 22, 2.6, M.phone, W / 2 + 0.3, 30, 0));
  g.add(box(1.2, 11, 2.6, M.phone, W / 2 + 0.3, 5, 0));
  return g;
}

// ---------- laptop ----------

function laptopScreenTexture() {
  return canvasTexture(1400, 840, (g, w, h) => {
    g.fillStyle = '#14161a'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#23262c'; g.fillRect(0, 0, w, 54);
    [['#ff5f57', 34], ['#febc2e', 66], ['#28c840', 98]].forEach(([c, x]) => {
      g.fillStyle = c; g.beginPath(); g.arc(x, 27, 10, 0, Math.PI * 2); g.fill();
    });
    g.fillStyle = '#9aa0a8'; g.font = '400 24px Menlo, Consolas, monospace';
    g.fillText('terminal — llama.cpp', 560, 35);

    g.font = '400 30px Menlo, Consolas, monospace';
    let y = 120;
    const line = (txt, c) => { g.fillStyle = c; y = wrapText(g, txt, 44, y, w - 88, 44); };
    line('$ ./llama-cli -m gemma-3n-E2B-it-Q4_K_M.gguf', '#e9e9e9');
    line('loaded model · 2.1 GB · CPU, 8 threads', '#7d8590');
    y += 18;
    line('> Ano ang mga sintomas ng dengue?', '#8eb4ff');
    y += 8;
    line('Kabilang sa mga karaniwang sintomas ang mataas na lagnat, matinding sakit ng ulo, pananakit ng kasukasuan at kalamnan, at pantal sa balat. Kumonsulta agad sa health center kung may pagdurugo o matinding pananakit ng tiyan.', '#e9e9e9');
    y += 18;
    line('[ 18.4 tok/s · offline ]', '#4ade80');
    g.fillStyle = '#e9e9e9'; g.fillText('> ', 44, y); g.fillRect(84, y - 26, 16, 32);
  });
}

function buildLaptop() {
  const g = new THREE.Group();
  const W = 304, D = 212, T = 13, L = 6;

  const base = slab(W, D, T, 9, M.brushed, 2);
  base.rotation.x = -Math.PI / 2;
  base.position.y = T / 2;
  g.add(base);

  const well = plate(276, 112, 4, M.well);
  well.rotation.x = -Math.PI / 2;
  well.position.set(0, T + 0.02, -36);
  g.add(well);

  const keyGeo = new THREE.BoxGeometry(16, 1.2, 15);
  const keys = new THREE.InstancedMesh(keyGeo, M.keys, 14 * 5);
  const m4 = new THREE.Matrix4();
  let k = 0;
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 14; c++) {
      keys.setMatrixAt(k++, m4.makeTranslation(-123.5 + c * 19, T + 0.6, -82 + r * 19));
    }
  }
  g.add(keys);
  g.add(box(112, 1.2, 15, M.keys, 0, T + 0.6, -82 + 5 * 19));
  for (const x of [-100, -80, 80, 100]) g.add(box(16, 1.2, 15, M.keys, x, T + 0.6, -82 + 5 * 19));

  const pad = plate(120, 72, 6, new THREE.MeshStandardMaterial({ color: 0xa9adb4, metalness: 0.6, roughness: 0.35 }));
  pad.rotation.x = -Math.PI / 2;
  pad.position.set(0, T + 0.02, 58);
  g.add(pad);

  // lid hinged at the back edge, opened ~110°
  const lid = new THREE.Group();
  const shell = slab(W, D, L, 9, M.brushed, 1.5);
  shell.position.set(0, D / 2, 0);
  lid.add(shell);
  const bezel = plate(W - 6, D - 6, 7, M.glass);
  bezel.position.set(0, D / 2, L / 2 + 0.01);
  lid.add(bezel);
  const display = new THREE.Mesh(new THREE.PlaneGeometry(W - 22, (W - 22) * 0.6),
    new THREE.MeshBasicMaterial({ map: laptopScreenTexture(), toneMapped: false }));
  display.position.set(0, D / 2 + 4, L / 2 + 0.03);
  lid.add(display);
  lid.position.set(0, T, -D / 2 + 4);
  lid.rotation.x = -0.35;
  g.add(lid);
  return g;
}

// ---------- scene plumbing ----------

function environment(renderer) {
  const env = new THREE.Scene();
  const room = new THREE.Mesh(new THREE.BoxGeometry(10, 10, 10),
    new THREE.MeshBasicMaterial({ color: 0x6a6a6a, side: THREE.BackSide }));
  env.add(room);
  const panel = (w, h, intensity, x, y, z, rx = 0, ry = 0) => {
    const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    mat.color.setScalar(intensity);
    const p = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    p.position.set(x, y, z); p.rotation.set(rx, ry, 0);
    env.add(p);
  };
  panel(6, 6, 5, 0, 4.9, 0, Math.PI / 2);
  panel(3, 5, 2.5, -4.9, 1, 0, 0, Math.PI / 2);
  panel(3, 5, 1.6, 4.9, 1, 1, 0, -Math.PI / 2);
  panel(5, 2, 2, 0, 1.5, 4.9);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const tex = pmrem.fromScene(env, 0.04).texture;
  pmrem.dispose();
  return tex;
}

function contactShadow() {
  const tex = canvasTexture(256, 256, (g, w, h) => {
    const grad = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    grad.addColorStop(0, 'rgba(26,26,26,0.32)');
    grad.addColorStop(1, 'rgba(26,26,26,0)');
    g.fillStyle = grad; g.fillRect(0, 0, w, h);
  });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 2.4),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
  m.rotation.x = -Math.PI / 2;
  return m;
}

const DEVICES = [
  { build: buildESP32,  elev: 0.62, phase: 0.4 },
  { build: buildPi5,    elev: 0.62, phase: 2.0 },
  { build: buildPhone,  elev: 0.2,  phase: 3.6 },
  { build: buildLaptop, elev: 0.38, phase: 5.0 },
];

function makeStage(spec, envTex) {
  const scene = new THREE.Scene();
  scene.environment = envTex;
  scene.add(new THREE.HemisphereLight(0xffffff, 0xd9d2c4, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(2, 4, 3);
  scene.add(key);

  const model = spec.build();
  const bounds = new THREE.Box3().setFromObject(model);
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
  model.position.sub(center);

  const fit = new THREE.Group();
  fit.scale.setScalar(1 / radius);
  fit.add(model);
  const spin = new THREE.Group();
  spin.add(fit);
  scene.add(spin);

  const shadow = contactShadow();
  const size = bounds.getSize(new THREE.Vector3());
  shadow.scale.setScalar(Math.max(size.x, size.z) / radius * 0.6);
  shadow.position.y = (bounds.min.y - center.y) / radius - 0.01;
  scene.add(shadow);

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  const dist = 1.08 / Math.sin(THREE.MathUtils.degToRad(15));
  camera.position.set(0, Math.sin(spec.elev) * dist, Math.cos(spec.elev) * dist);
  camera.lookAt(0, 0, 0);
  return { scene, camera, spin, phase: spec.phase };
}

function init(root) {
  const canvas = root.querySelector('canvas');
  const cells = [...root.querySelectorAll('.hw-stage')];
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch (e) {
    root.classList.add('no-webgl');
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.setClearColor(0x000000, 0);

  const envTex = environment(renderer);
  const stages = DEVICES.map((d) => makeStage(d, envTex));
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) ||
        canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false);
    }
  }

  function render(t) {
    resize();
    const cr = canvas.getBoundingClientRect();
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setScissorTest(true);
    stages.forEach((s, i) => {
      const r = cells[i].getBoundingClientRect();
      const x = r.left - cr.left, y = cr.bottom - r.bottom;
      renderer.setViewport(x, y, r.width, r.height);
      renderer.setScissor(x, y, r.width, r.height);
      s.camera.aspect = r.width / r.height;
      s.camera.updateProjectionMatrix();
      s.spin.rotation.y = s.phase + (reduced ? 0 : t * 0.00035);
      renderer.render(s.scene, s.camera);
    });
  }

  let visible = false, raf = 0;
  const loop = (t) => { render(t); if (visible && !reduced) raf = requestAnimationFrame(loop); };
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    cancelAnimationFrame(raf);
    if (visible) raf = requestAnimationFrame(loop);
  }).observe(root);
  window.addEventListener('resize', () => render(performance.now()));
  root.classList.add('webgl-ready');
}

const root = document.getElementById('fig-hardware');
if (root) init(root);
