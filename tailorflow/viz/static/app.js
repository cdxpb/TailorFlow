import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/* ------------------------------------------------------------------ api */
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const statusText = $('#statusText'), spinner = $('.spin');
const busyBar = $('#busyBar'), busyLabel = $('#busyLabel'), busyTime = $('#busyTime'), busyCancel = $('#busyCancel');
let busy = 0;              // in-flight requests
let flows = 0;             // user flows (a sample batch is many requests; the lock spans all of them)
let busyOff = null, busySince = 0, busyTick = null, wasBusy = false;
/* Cancellation. One AbortController covers the flow that is running; the X on the banner aborts the request in flight
   and sets `cancelReq`, so every later call() in that flow's loop throws immediately and the flow unwinds through its
   own finally (which is where pending cards are cleaned up). Aborting the fetch does not stop the server's GPU work -
   the result is simply dropped, which is the safe behaviour (the server stays consistent and the next call reuses it). */
let abortCtl = null, cancelReq = false;
function beginScope() { if (flows === 0) { abortCtl = new AbortController(); cancelReq = false; } }
function endScope() { abortCtl = null; cancelReq = false; busyCancel.hidden = true; }
function cancelFlow() {
  if (!isBusy()) return;
  cancelReq = true;
  busyCancel.disabled = true;
  say('stopping…');
  abortCtl?.abort();
}

/* The whole app must look and behave as one blocked unit while anything runs: a sample batch is a dozen sequential
   requests with renders in between, so a request counter alone would flick the UI back to interactive mid-flow.
   `flows` is held by the safe() wrapper around every handler, `busy` by the request itself. */
function busySync() {
  const on = flows > 0 || busy > 0;
  const started = on && !wasBusy;
  wasBusy = on;
  // inert is stronger than pointer-events: it also removes the panels from tab order while a flow runs
  for (const el of ['header', '#tools', '#preview', '#bodyBar']) {
    const n = document.querySelector(el);
    if (n) n.toggleAttribute('inert', on);
  }
  if (on) {
    clearTimeout(busyOff); busyOff = null;
    document.body.classList.add('busy');
    busyBar.hidden = false;
    if (started) { busySince = performance.now(); busyLabel.textContent = 'working…'; }   // never show the last flow's result
    if (!busySince) busySince = performance.now();
    busyCancel.hidden = false;
    busyCancel.disabled = cancelReq;
    if (!busyTick) busyTick = setInterval(() => {
      if (busySince) busyTime.textContent = ((performance.now() - busySince) / 1000).toFixed(1) + ' s';
    }, 100);
  } else {
    clearTimeout(busyOff);
    busyOff = setTimeout(() => {
      if (flows > 0 || busy > 0) return;
      clearInterval(busyTick); busyTick = null; busySince = 0;
      document.body.classList.remove('busy');
      busyBar.hidden = true; busyTime.textContent = '';
    }, 180);
  }
} 
const isBusy = () => flows > 0 || busy > 0 || document.body.classList.contains('busy');

async function call(path, body) {
  if (cancelReq) throw new DOMException('cancelled', 'AbortError');   // a later step of a cancelled flow must not start
  busy++; busySync(); $('#status').classList.add('busy'); spinner.hidden = false;
  try {
    const r = await fetch(path, {
      ...(body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
      signal: abortCtl ? abortCtl.signal : undefined,
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const buf = await r.arrayBuffer();
    const n = new DataView(buf).getUint32(0, true);
    const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
    const base = 4 + n, arrays = {};
    for (const [k, d] of Object.entries(meta.arrays || {})) {
      const T = d.dtype === 'uint32' ? Uint32Array : Float32Array;
      arrays[k] = new T(buf.slice(base + d.offset, base + d.offset + d.bytes));
    }
    if (meta.error) throw new Error(meta.error);
    return { meta, arrays };
  } finally {
    if (--busy === 0) { $('#status').classList.remove('busy'); spinner.hidden = true; }
    busySync();
  }
}

const say = (t) => {
  statusText.textContent = t;
  if (!busyBar.hidden) busyLabel.textContent = t;
};
/* Every user-facing handler goes through safe(): it catches errors into the status line AND holds the busy lock for the
   duration of the whole flow, so buttons stay dead from the first click to the last render of a batch. The outermost
   safe() owns the cancellation scope, so one X stops the sample batch / mesh-all / transfer currently running. */
const safe = (fn) => (...a) => {
  const outer = flows === 0;
  beginScope();
  const r = (async () => {
    if (!isBusy()) { say('working…'); }          // name the action immediately, before the first round-trip
    flows++; busySync();
    try { return await fn(...a); }
    catch (e) {
      if (e.name === 'AbortError' || cancelReq) say('stopped');
      else say('⚠ ' + e.message);
    }
    finally { flows--; if (outer) endScope(); busySync(); }
  })();
  return r;
};
async function timed(label, fn) {
  say(label + '…');
  const t0 = performance.now();
  const out = await fn();
  say(`${label} · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  return out;
}
const rseed = () => Math.floor(Math.random() * 1e6);
const MESH_PRESET = 'b3x_lap30';   // the only preset the UI uses (the server's default)
const GRAV = -9.81, PHYS_DT = 1 / 60, PHYS_SUB = 3;   // PHYS_DT/SUB is the solver step

/* ---------------------------------------------------------------- state */
const LOOK_KEY = 'tailorflow.look';
const LOOK_DEFAULT = { theme: 'light', color: '#5b7cfa', finish: 'satin', wire: false, bodyOpacity: 30 };
const FINISH = {
  matte: { roughness: 0.95, metalness: 0.0 },
  satin: { roughness: 0.55, metalness: 0.06 },
  glossy: { roughness: 0.22, metalness: 0.18 },
  metal: { roughness: 0.32, metalness: 0.9 },
};
const SWATCHES = ['#5b7cfa', '#8b5cf6', '#ec4899', '#ef4444', '#f59e0b', '#10b981', '#06b6d4', '#64748b'];
const THEME = {
  light: { bg: 0xe9edf4, body: 0xb4bac7, bare: 0xd7dbe3, hemi: [0xffffff, 0xd3d8e4, 2.0], key: 1.5, rim: 0.5 },
  dark: { bg: 0x0c0d10, body: 0x767d8f, bare: 0x3a3f4b, hemi: [0xbcc6e0, 0x191c22, 2.1], key: 1.5, rim: 0.7 },
};

const S = {
  init: null,
  body: { space: 0, beta: new Float32Array(10), verts: null },     // the body under the viewport when no card is active
  occ: null, occBase: null,                                         // painted mask; the active garment's own coverage
  items: new Map(), sets: { samples: [], transfer: [] }, set: 'samples', active: null,
  tab: 'sample', cls: null, n: 4, tn: 4, tspace: 'mixed', solver: 'heun', seedLock: false,
  view: { show: 'mesh', color: 'plain', body: true },   // 'mesh' falls back to points until a mesh exists
  look: { ...LOOK_DEFAULT }, paint: false, brush: 1,
  phys: { on: false },
};
try { Object.assign(S.look, JSON.parse(localStorage.getItem(LOOK_KEY) || '{}')); } catch { /* private mode */ }

const curL = () => +$('#level').value;
const solverNow = () => S.solver;
const cfgNow = () => (curL() > 4096 ? 1.0 : +$('#cfg').value);
const act = () => (S.active ? S.items.get(S.active) : null);
const title = (it) => `${it.meta.cls} #${it.meta.key.slice(1)}`;
function subtitle(it) {
  const m = it.meta, st = m.stats;
  const who = m.kind === 'transfer' ? m.body.space_name : `seed ${m.seed}`;
  return `${who} · ${st.inside.toFixed(1)}% in · ${(st.n / 1024).toFixed(0)}k`;
}

/* ----------------------------------------------------------------- 3-D */
const canvas = $('#gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
camera.position.set(0, 0.15, 3.1);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.target.set(0, 0.1, 0);

function lights(sc) {
  const hemi = new THREE.HemisphereLight(0xffffff, 0xd3d8e4, 2.0);
  const key = new THREE.DirectionalLight(0xffffff, 1.5); key.position.set(1.4, 2.2, 1.8);
  const rim = new THREE.DirectionalLight(0xb9c9ff, 0.5); rim.position.set(-1.6, 0.6, -1.4);
  sc.add(hemi, key, rim);
  return { hemi, key, rim };
}
function makeCell(sc) {
  const body = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({
    roughness: 0.85, metalness: 0, transparent: true, depthWrite: false, vertexColors: true,
  }));
  const pts = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 0.004, vertexColors: true }));
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ side: THREE.DoubleSide }));
  pts.visible = mesh.visible = false;
  sc.add(body, pts, mesh);
  return { body, pts, mesh, L: lights(sc), sc };
}
const main = makeCell(scene);
main.mesh.frustumCulled = false;   // physics moves vertices without updating the bounding sphere

// thumbnails: a second, small renderer draws each card once, front view, into an <img>
const thumbCanvas = document.createElement('canvas');
const TW = 240, TH = 320;
const thumbR = new THREE.WebGLRenderer({ canvas: thumbCanvas, antialias: true, preserveDrawingBuffer: true });
thumbR.setPixelRatio(1); thumbR.setSize(TW, TH, false);
const tScene = new THREE.Scene();
const tCam = new THREE.PerspectiveCamera(26, TW / TH, 0.01, 100);
const thumb = makeCell(tScene);

const garmentColor = new THREE.Color(S.look.color);
const bareColor = new THREE.Color(THEME[S.look.theme].bare);
const coverColor = new THREE.Color(0xe8850c);
const badColor = new THREE.Color(0xd63b3f);

/* A mesh only "exists" when it has both vertices AND faces. The server can legitimately return a vertex buffer with
   zero faces (every triangle trimmed away by the rim/component filter); an empty Uint32Array is truthy in JS, so the
   naive `!!it.mverts` test would hide the pointcloud and draw an index-less geometry that renders as nothing - the
   garment becomes invisible in both the viewport and the card. Every presence test goes through here. */
const meshOK = (it) => !!(it && it.mverts && it.mverts.length && it.mfaces && it.mfaces.length);
/* A mesh can be present, indexed and still unrenderable: NaN vertices (a divergent clean-up pass) or every vertex at
   one point (a collapsed iso-surface) both give a mesh that draws nothing. Reject those so the pointcloud shows. */
function meshFinite(V, F) {
  if (!V || !V.length) return false;
  const n = V.length / 3;
  if (F && F.length) { for (let i = 0; i < F.length; i++) if (F[i] >= n) return false; }
  let lo0 = Infinity, lo1 = Infinity, lo2 = Infinity, hi0 = -Infinity, hi1 = -Infinity, hi2 = -Infinity;
  for (let i = 0; i < V.length; i += 3) {
    const x = V[i], y = V[i + 1], z = V[i + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return false;
    if (x < lo0) lo0 = x; if (x > hi0) hi0 = x;
    if (y < lo1) lo1 = y; if (y > hi1) hi1 = y;
    if (z < lo2) lo2 = z; if (z > hi2) hi2 = z;
  }
  return (hi0 - lo0) + (hi1 - lo1) + (hi2 - lo2) > 1e-5;
}

/* ------------------------------------------------------------ geometry */
function setGeom(geo, pos) {
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.computeBoundingSphere();
}
function setBody(cell, verts) {
  const g = cell.body.geometry;
  setGeom(g, verts);
  if (!g.index) g.setIndex(new THREE.BufferAttribute(S.init.faces, 1));
  if (!g.getAttribute('color')) g.setAttribute('color', new THREE.BufferAttribute(new Float32Array((verts.length / 3) * 4), 4));
  g.computeVertexNormals();
}
function setMesh(cell, verts, faces) {
  const g = cell.mesh.geometry;
  if (!verts || !verts.length || !faces || !faces.length) { g.setIndex(null); return; }
  setGeom(g, verts);
  g.setIndex(new THREE.BufferAttribute(faces, 1));
  g.computeVertexNormals();
}

/* ---------------------------------------------------------- cloth physics
   A stable displacement-field cloth. Every vertex keeps a small offset `disp` from the mesh it was decoded
   at (`rest`), and the body is represented by a spring pulling that offset back to zero. Gravity sags the
   sheet a few centimetres, an apparent-wind term fed by the camera's angular velocity pushes it around when
   you spin the view, and a Laplacian pass over the mesh edges couples neighbouring offsets so the sheet
   ripples as one piece instead of every vertex jittering on its own. The offset is hard-clamped to a small
   fraction of the garment height, so the cloth can never leave the body no matter how fast the view moves. */
const PHYS = {
  armed: false, n: 0, rest: null, pos: null, disp: null, vel: null,
  anchor: null, ea: null, eb: null, acc: null, cnt: null, h: 1, maxDisp: 0.04, frame: 0,
};
const P_GRAV = -9.81, P_K = 300, P_DAMP = 0.98, P_LAM = 0.5, P_WIND = 6, P_VMAX = 4;

function physBuildEdges(faces) {
  const seen = new Set(), ea = [], eb = [];
  for (let f = 0; f + 2 < faces.length; f += 3) {
    const t = [faces[f], faces[f + 1], faces[f + 2]];
    for (let k = 0; k < 3; k++) {
      const i = t[k], j = t[(k + 1) % 3];
      if (i === j) continue;
      const key = i < j ? i * PHYS.n + j : j * PHYS.n + i;
      if (seen.has(key)) continue;
      seen.add(key); ea.push(i); eb.push(j);
    }
  }
  PHYS.ea = Int32Array.from(ea); PHYS.eb = Int32Array.from(eb);
}

function physInit(verts, faces) {
  const n = verts.length / 3;
  PHYS.n = n;
  PHYS.rest = Float32Array.from(verts);
  PHYS.pos = Float32Array.from(verts);
  PHYS.disp = new Float32Array(n * 3);
  PHYS.vel = new Float32Array(n * 3);
  PHYS.anchor = new Uint8Array(n);
  PHYS.acc = new Float32Array(n * 3);
  PHYS.cnt = new Uint32Array(n);
  let mnY = Infinity, mxY = -Infinity;
  for (let i = 0; i < n; i++) { const y = verts[i * 3 + 1]; if (y < mnY) mnY = y; if (y > mxY) mxY = y; }
  const h = Math.max(1e-3, mxY - mnY);
  PHYS.h = h;
  PHYS.maxDisp = h * 0.05;                     // the cloth may never leave the body by more than 5% of its height
  for (let i = 0; i < n; i++) PHYS.anchor[i] = (verts[i * 3 + 1] - mnY) / h < 0.06 ? 1 : 0;   // pin the top band
  physBuildEdges(faces);
  PHYS.armed = true;
}

function physStep(dt, w, cx, cy, cz) {
  const { pos, rest, disp, vel, anchor, n } = PHYS;
  if (!disp) return;
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    if (anchor[i]) { disp[o] = disp[o + 1] = disp[o + 2] = 0; vel[o] = vel[o + 1] = vel[o + 2] = 0; continue; }
    // apparent wind: -omega x (p - centre), strongest at the silhouette, which is what makes a quick spin ripple
    const rx = rest[o] - cx, ry = rest[o + 1] - cy, rz = rest[o + 2] - cz;
    const ax = -(w[1] * rz - w[2] * ry) * P_WIND;
    const ay = -(w[2] * rx - w[0] * rz) * P_WIND + P_GRAV;
    const az = -(w[0] * ry - w[1] * rx) * P_WIND;
    let vx = vel[o] + ax * dt - P_K * disp[o] * dt;              // semi-implicit Euler, stiff spring = heavy cloth
    let vy = vel[o + 1] + ay * dt - P_K * disp[o + 1] * dt;
    let vz = vel[o + 2] + az * dt - P_K * disp[o + 2] * dt;
    vx *= P_DAMP; vy *= P_DAMP; vz *= P_DAMP;
    const sp = Math.hypot(vx, vy, vz);
    if (sp > P_VMAX) { const s = P_VMAX / sp; vx *= s; vy *= s; vz *= s; }
    vel[o] = vx; vel[o + 1] = vy; vel[o + 2] = vz;
    disp[o] += vx * dt; disp[o + 1] += vy * dt; disp[o + 2] += vz * dt;
  }
  const { acc, cnt, ea, eb } = PHYS;
  acc.fill(0); cnt.fill(0);
  for (let k = 0; k < ea.length; k++) {
    const i = ea[k] * 3, j = eb[k] * 3;
    acc[i] += disp[j]; acc[i + 1] += disp[j + 1]; acc[i + 2] += disp[j + 2]; cnt[ea[k]]++;
    acc[j] += disp[i]; acc[j + 1] += disp[i + 1]; acc[j + 2] += disp[i + 2]; cnt[eb[k]]++;
  }
  const maxD = PHYS.maxDisp, maxD2 = maxD * maxD;
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    if (anchor[i]) continue;
    if (cnt[i]) {                                                // pull each offset toward its neighbours'
      const c = cnt[i];
      disp[o] += P_LAM * (acc[o] / c - disp[o]);
      disp[o + 1] += P_LAM * (acc[o + 1] / c - disp[o + 1]);
      disp[o + 2] += P_LAM * (acc[o + 2] / c - disp[o + 2]);
    }
    const d2 = disp[o] * disp[o] + disp[o + 1] * disp[o + 1] + disp[o + 2] * disp[o + 2];
    if (d2 > maxD2) { const s = maxD / Math.sqrt(d2); disp[o] *= s; disp[o + 1] *= s; disp[o + 2] *= s; }
  }
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    pos[o] = rest[o] + disp[o]; pos[o + 1] = rest[o + 1] + disp[o + 1]; pos[o + 2] = rest[o + 2] + disp[o + 2];
  }
}

// camera angular velocity (world space) -> apparent wind, then a few substeps per frame
const _qPrev = new THREE.Quaternion(), _qNow = new THREE.Quaternion(), _dq = new THREE.Quaternion();
const _axis = new THREE.Vector3();
let _lastT = 0, _qReady = false;
function tickPhysics(now) {
  const dt = Math.min(0.033, Math.max(0.001, (now - _lastT) / 1000));
  _lastT = now;
  if (!S.phys.on || !PHYS.armed || !PHYS.pos || !main.mesh.visible) { _qReady = false; return; }
  camera.getWorldQuaternion(_qNow);
  let w = [0, 0, 0];
  if (_qReady) {
    _dq.copy(_qPrev).invert().multiply(_qNow);
    const s = Math.hypot(_dq.x, _dq.y, _dq.z);
    if (s > 1e-5) {
      const ang = 2 * Math.atan2(s, Math.abs(_dq.w));
      _axis.set(_dq.x, _dq.y, _dq.z).multiplyScalar((_dq.w < 0 ? -1 : 1) / s);
      const rate = Math.min(4, ang / dt);                        // clamp so a flick of the mouse cannot explode it
      w = [_axis.x * rate, _axis.y * rate, _axis.z * rate];
    }
  }
  _qPrev.copy(_qNow); _qReady = true;
  const c = controls.target;
  const sub = PHYS_DT / PHYS_SUB;
  for (let s = 0; s < PHYS_SUB; s++) physStep(sub, w, c.x, c.y, c.z);
  physSync();
}

function physSync() {
  const g = main.mesh.geometry, attr = g.getAttribute('position');
  if (!attr || attr.array !== PHYS.pos) return;                      // only the viewport reads the simulated buffer
  attr.needsUpdate = true;
  if ((PHYS.frame++ & 3) === 0) g.computeVertexNormals();            // lighting lags a few frames; the fold moves little
}

function physArm(it) {
  if (!S.phys.on || !meshOK(it)) { PHYS.armed = false; return; }
  physInit(it.mverts, it.mfaces);
  // render straight from the simulated buffer, so the item's own mesh data is never mutated
  const g = main.mesh.geometry;
  g.setAttribute('position', new THREE.BufferAttribute(PHYS.pos, 3));
  _lastT = 0; _qReady = false;
}
function physReset() {
  if (!PHYS.pos || !PHYS.rest) return;
  PHYS.disp.fill(0); PHYS.vel.fill(0);
  PHYS.pos.set(PHYS.rest);
  physSync();
}
function cloudColors(n, signed, mode) {
  const col = new Float32Array(n * 3);
  const tmp = new THREE.Color();
  for (let i = 0; i < n; i++) {
    let c = garmentColor;
    if (mode === 'depth' && signed && signed[i] < 0) c = tmp.copy(garmentColor).lerp(badColor, Math.min(1, -signed[i] / 12));
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
  }
  return col;
}
/* Point sprite size in world units. Sparse clouds have wide spacing, so they need visibly larger dots; dense clouds
   stay close to their spacing so the surface still reads as cloth rather than a solid blob. */
function cloudSize(n) {
  if (n <= 4096) return 0.0125;
  if (n <= 8192) return 0.0105;
  if (n <= 16384) return 0.0085;
  return 0.0065;
}
function setCloud(cell, pts, signed, mode) {
  if (!pts) { cell.pts.visible = false; return; }
  setGeom(cell.pts.geometry, pts);
  cell.pts.geometry.setAttribute('color', new THREE.BufferAttribute(cloudColors(pts.length / 3, signed, mode), 3));
  cell.pts.material.size = cloudSize(pts.length / 3);
}
function overlayOcc() {                                            // the occupancy the body should show, if any
  if (S.paint) return S.occ;                                       // painting: the working mask
  const it = act();
  if (it && it.occ && S.tab === 'occ') return it.occ;              // editing: always show the garment's own coverage
  if (!$('#useOcc').checked) return null;                          // otherwise the toggle is the only authority
  return it && it.occ ? it.occ : S.occ;
}
function writeBodyColors(cell, occ, base, baseA, coverA) {          // body colour + alpha live in the vertex colours
  const c = cell.body.geometry.getAttribute('color');
  if (!c) return;
  for (let i = 0; i < c.count; i++) {
    const on = !!occ && occ[i] > 0.5, col = on ? coverColor : base, a = on ? coverA : baseA;
    if (c.itemSize === 4) c.setXYZW(i, col.r, col.g, col.b, a);
    else c.setXYZ(i, col.r, col.g, col.b);
  }
  c.needsUpdate = true;
}
function bodyColors() {
  const th = THEME[S.look.theme];
  const baseA = S.paint ? 1 : S.look.bodyOpacity / 100;
  return { base: new THREE.Color(S.paint ? th.bare : th.body), baseA, coverA: Math.min(1, Math.max(baseA, 0.55)) };
}
function applyBodyOverlay() {
  const { base, baseA, coverA } = bodyColors();
  writeBodyColors(main, overlayOcc(), base, baseA, coverA);
}
function styleCell(cell) {
  const th = THEME[S.look.theme], f = FINISH[S.look.finish] || FINISH.satin;
  cell.sc.background = new THREE.Color(th.bg);
  cell.L.hemi.color.setHex(th.hemi[0]); cell.L.hemi.groundColor.setHex(th.hemi[1]); cell.L.hemi.intensity = th.hemi[2];
  cell.L.key.intensity = th.key; cell.L.rim.intensity = th.rim;
  cell.body.material.color.set(0xffffff);                          // body colour + alpha are per-vertex (setBody/applyBodyOverlay)
  cell.body.material.opacity = 1;
  cell.mesh.material.color.copy(garmentColor);
  cell.mesh.material.roughness = f.roughness;
  cell.mesh.material.metalness = f.metalness;
  cell.mesh.material.wireframe = S.look.wire;
  cell.mesh.material.needsUpdate = true;
}

/* ------------------------------------------------------- viewport */
function frameView(margin = 1.12) {
  const g = main.body.geometry;
  if (!g.attributes.position) return;
  const bs = g.boundingSphere, r = bs.radius * margin;
  const fovV = THREE.MathUtils.degToRad(camera.fov);
  const fovH = 2 * Math.atan(Math.tan(fovV / 2) * camera.aspect);
  const dist = r / Math.sin(Math.min(fovV, fovH) / 2);
  const dir = camera.position.clone().sub(controls.target).normalize();
  if (!isFinite(dir.x)) dir.set(0, 0, 1);
  controls.target.copy(bs.center);
  camera.position.copy(bs.center).addScaledVector(dir, dist);
  camera.near = Math.max(0.01, dist - r * 2); camera.far = dist + r * 4;
  camera.updateProjectionMatrix(); controls.update();
}
function applyView() {
  const v = S.view, it = act(), overlay = overlayOcc();
  const hasMesh = meshOK(it);
  main.pts.visible = !!it && (S.paint || v.show !== 'mesh' || !hasMesh);
  main.mesh.visible = !S.paint && hasMesh && v.show !== 'points';
  const showPhys = hasMesh && !S.paint && main.mesh.visible;
  $('#physBtn').hidden = !showPhys;
  $('#physBtn').classList.toggle('on', !!S.phys.on && showPhys);
  main.pts.material.transparent = S.paint;
  main.pts.material.opacity = S.paint ? 0.3 : 1;
  main.pts.material.needsUpdate = true;
  main.body.visible = v.body || S.paint || !!overlay;               // the Body toggle owns the body, the overlay only paints on it
  main.body.material.opacity = 1;                                  // body alpha lives in the vertex colours
  main.body.material.depthWrite = S.paint;
  main.body.material.vertexColors = true;
  main.body.material.color.set(0xffffff);
  main.body.material.needsUpdate = true;
  applyBodyOverlay();
}
function showActive(frame = false) {
  const it = act();
  const verts = it ? it.verts : S.body.verts;
  if (verts) setBody(main, verts);
  setCloud(main, it?.points, it?.signed, S.view.color);
  setMesh(main, it?.mverts, it?.mfaces);
  physArm(it);
  applyView();
  if (frame) frameView();
  stats(); bodyBar(); targets();
  $('#viewTitle').innerHTML = it
    ? `<b>${title(it)}</b> · ${it.meta.kind === 'transfer' ? `transfer of #${it.meta.parent.slice(1)}` : it.meta.parent ? `edit of #${it.meta.parent.slice(1)}` : 'sample'}`
    : 'no garment · <b>Sample</b> to start';
}
function stats() {
  const it = act(), st = it?.meta.stats, m = it?.meta;
  $('#stats').innerHTML = !it ? '' : [
    ['points', st.n.toLocaleString()],
    ['spacing', st.spacing_mm + ' mm'],
    ['inside', st.inside + ' %', st.inside > 3 ? 'bad' : st.inside > 1 ? 'warn' : ''],
    ['coverage', m.cond ? `${m.occ_src} · ${m.covered} v` : 'free'],
    m.mesh && meshOK(it) && ['mesh', `${m.mesh.verts.toLocaleString()} v · ${m.mesh.loops} loops`],
  ].filter(Boolean).map(([k, v, cls]) => `<span class="pill ${cls || ''}">${k} <b>${v}</b></span>`).join('');
}
function layout() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
function draw(now = 0) {
  controls.update();
  tickPhysics(now);
  renderer.render(scene, camera);
  requestAnimationFrame(draw);
}
new ResizeObserver(layout).observe(canvas);

/* ------------------------------------------------------------ thumbnails */
function robustBox(p, q = 0.01) {                                  // 1-99 % box: stray points don't shrink the thumbnail
  const n = p.length / 3, stride = Math.max(1, Math.floor(n / 4000)), lo = [], hi = [];
  for (let a = 0; a < 3; a++) {
    const v = [];
    for (let i = 0; i < n; i += stride) v.push(p[i * 3 + a]);
    v.sort((x, y) => x - y);
    lo.push(v[Math.floor(q * (v.length - 1))]); hi.push(v[Math.ceil((1 - q) * (v.length - 1))]);
  }
  return { c: new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2),
           sz: new THREE.Vector3(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) };
}
function renderThumb(it) {
  setBody(thumb, it.verts);
  const { base, baseA, coverA } = bodyColors();
  writeBodyColors(thumb, null, base, baseA, coverA);
  thumb.body.visible = S.look.bodyOpacity > 0;
  const useMesh = meshOK(it) && S.view.show !== 'points';
  setMesh(thumb, useMesh ? it.mverts : null, it.mfaces);
  thumb.mesh.visible = useMesh;
  thumb.pts.visible = !useMesh;
  if (!useMesh) { setCloud(thumb, it.points, it.signed, S.view.color); thumb.pts.material.size = cloudSize(it.points.length / 3) * 1.15; }
  const { c, sz } = robustBox(useMesh ? it.mverts : it.points);
  const half = Math.max(sz.y / 2, sz.x / 2 / tCam.aspect) * 1.3;      // garment fills the card, 3:4
  const dist = half / Math.tan(THREE.MathUtils.degToRad(tCam.fov) / 2) + sz.z / 2;
  tCam.position.set(c.x + dist * 0.12, c.y + dist * 0.05, c.z + dist);
  tCam.lookAt(c);
  tCam.near = 0.01; tCam.far = dist + 3; tCam.updateProjectionMatrix();
  thumbR.render(tScene, tCam);
  it.thumb = thumbCanvas.toDataURL('image/png');
}
function rethumbAll() {
  styleCell(thumb);
  for (const it of S.items.values()) if (!it.pending) renderThumb(it);
  renderCards(); targets();
}

/* ------------------------------------------------------------- cards */
function renderCards() {
  const keys = S.sets[S.set];
  $('#cards').innerHTML = '';
  for (const k of keys) {
    const it = S.items.get(k);
    const el = document.createElement('div');
    el.className = 'pcard' + (it.pending ? ' pending' : '') + (k === S.active ? ' sel' : '');
    el.dataset.key = k;
    if (it.pending) {
      el.innerHTML = `<div class="thumb"></div><div class="meta"><b>${it.label}</b><span>sampling…</span></div>`;
    } else {
      const m = it.meta, tags = [];
      if (m.cond) tags.push(`<em class="cov" title="${m.occ_src}">${m.occ_src === 'edited' ? 'edit' : 'cov'}</em>`);
      if (m.mesh && meshOK(it)) tags.push('<em class="mesh">mesh</em>');
      el.innerHTML = `<div class="thumb"><img alt="" src="${it.thumb}"></div><div class="tags">${tags.join('')}</div>`
        + `<button class="x" title="remove">×</button><div class="meta"><b>${title(it)}</b><span>${subtitle(it)}</span></div>`;
      el.onclick = (e) => {
        if (e.target.classList.contains('x')) { forget([k]); return; }
        pickCard(k);
      };
    }
    $('#cards').appendChild(el);
  }
  $('#emptyNote').hidden = keys.length > 0;
  $('#emptyNote').textContent = S.set === 'samples'
    ? 'Samples appear here. Pick one to inspect it in the viewport.'
    : 'Transfers appear here: one garment decoded on several bodies.';
  $('#clearSet').disabled = !keys.length;
  syncSegs(); syncMesh();
}
function select(key, frame = false) {
  if (S.paint && key !== S.active) setPaint(false);
  const prevBody = act()?.verts || S.body.verts;
  S.active = key;
  const it = act();
  if (it) { S.body.space = it.meta.body.space; S.body.beta = Float32Array.from(it.meta.body.beta); S.body.verts = it.verts; }
  S.occBase = null; updateCoverageHint();
  showActive(frame || !prevBody);
  $$('#cards .pcard').forEach((el) => el.classList.toggle('sel', el.dataset.key === key));
  $('#cards .pcard.sel')?.scrollIntoView({ block: 'nearest' });
}
// clicking a card is an edit gesture: select it and open the Edit pane on it
function pickCard(key) {
  setTab('occ', true);
  select(key);
}
function step(d) {
  const keys = S.sets[S.set].filter((k) => !S.items.get(k).pending);
  if (!keys.length) return;
  const i = keys.indexOf(S.active);
  pickCard(keys[(i + d + keys.length) % keys.length]);
}
function showSet(name) { S.set = name; renderCards(); }
function clearActive() {                                            // back to an empty body, no card selected
  if (S.paint) setPaint(false);
  S.active = null; S.occBase = null;
  showActive(); renderCards(); updateCoverageHint();
}
async function forget(keys) {
  keys = keys.filter((k) => !S.items.get(k)?.pending);
  if (!keys.length) return;
  await call('/api/forget', { keys });
  for (const k of keys) S.items.delete(k);
  for (const s of Object.keys(S.sets)) S.sets[s] = S.sets[s].filter((k) => !keys.includes(k));
  if (keys.includes(S.active)) {
    const next = S.sets[S.set][0] || S.sets.samples[0] || S.sets.transfer[0] || null;
    S.active = null;
    if (next) select(next); else showActive();
  }
  renderCards();
}
function ingest(meta, arrays, prefix = '') {
  // the server sends an empty payload array for "no mesh"; turn that into null so every presence test is honest
  const arr = (k) => { const a = arrays[prefix + k]; return a && a.length ? a : null; };
  let mverts = arr('mverts'), mfaces = arr('mfaces');
  if (mverts && !mfaces) { mverts = null; }                 // a vertex soup is not a mesh
  if (mverts && !meshFinite(mverts, mfaces)) { mverts = null; mfaces = null; }   // NaN/zero-extent: keep the pointcloud
  const it = {
    meta, points: arrays[prefix + 'points'], signed: arrays[prefix + 'signed'], verts: arrays[prefix + 'verts'],
    occ: arr('occ'),
    mverts, mfaces,
  };
  S.items.set(meta.key, it);
  renderThumb(it);
  return it;
}
function replaceKey(set, oldKey, newKey) {
  const i = S.sets[set].indexOf(oldKey);
  if (i >= 0) S.sets[set][i] = newKey; else S.sets[set].push(newKey);
}

/* ------------------------------------------------------------- flows */
async function doSample() {
  const cond = $('#useOcc').checked ? 'painted' : 'none';
  if (cond === 'painted' && coveredCount() === 0) throw new Error('paint or randomize a coverage first');
  if (S.paint) setPaint(false);
  const n = S.n, base = S.seedLock ? +$('#seed').value : rseed();
  $('#seed').value = base;
  const old = $('#appendSamples').checked ? [] : S.sets.samples.slice();
  if (old.length) await forget(old);
  const ph = [];
  for (let i = 0; i < n; i++) {
    const k = `pending${rseed()}`;
    S.items.set(k, { pending: true, label: `${S.cls} · ${i + 1}/${n}` });
    S.sets.samples.push(k); ph.push(k);
  }
  showSet('samples');
  const q = {
    cls: S.cls, L: curL(), solver: solverNow(), steps: +$('#steps').value, cfg: cfgNow(), cond,
    body: { space: S.body.space, beta: Array.from(S.body.beta) },
  };
  const t0 = performance.now();
  let first = true;
  try {
    for (let i = 0; i < n; i++) {
      say(`Sampling ${i + 1}/${n}…`);
      const { meta, arrays } = await call('/api/sample', { ...q, seed: base + i });
      ingest(meta.item, arrays);
      S.items.delete(ph[i]); replaceKey('samples', ph[i], meta.item.key);
      renderCards();
      if (first) { first = false; select(meta.item.key, true); setTab('occ', true); }   // inspect it as soon as it lands
    }
    say(`${n} sample${n > 1 ? 's' : ''} · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  } finally {
    for (const k of ph) if (S.items.get(k)?.pending) { S.items.delete(k); S.sets.samples = S.sets.samples.filter((x) => x !== k); }
    renderCards();
  }
}

async function updateItem(meta, arrays) {
  const it = ingest(meta.item, arrays);
  if (S.active === meta.item.key) showActive();
  renderCards();
  return it;
}
async function doMesh(key = S.active) {
  const it = S.items.get(key);
  if (!it || it.pending) throw new Error('pick a garment first');
  if (meshOK(it)) return it.meta.mesh || null;                                // already meshed
  const { meta, arrays } = await call('/api/mesh', { key, preset: MESH_PRESET });
  await updateItem(meta, arrays);
  if (!meshOK(S.items.get(key))) say(`⚠ mesh of ${title(it)} came back empty - keeping the pointcloud`);
  return meta.item.mesh;
}
async function doMeshAll() {
  const keys = meshable();
  if (!keys.length) return;
  for (let i = 0; i < keys.length; i++) {
    say(`Meshing ${i + 1}/${keys.length}…`);
    await doMesh(keys[i]);
  }
  if (S.view.show === 'points') { S.view.show = 'mesh'; applyView(); rethumbAll(); syncSegs(); }
  say(`meshed ${keys.length} card${keys.length === 1 ? '' : 's'}`);
}
async function doTransfer() {
  const it = act();
  if (!it) throw new Error('pick a garment first');
  const src = it.meta.key;
  const space = S.tspace === 'mixed' ? null : +S.tspace;
  const { meta, arrays } = await timed(`Transferring to ${S.tn} bodies`,
    () => call('/api/transfer', { key: src, n: S.tn, space, seed: rseed() }));
  const old = S.sets.transfer.slice();
  if (old.length) { await call('/api/forget', { keys: old }); old.forEach((k) => S.items.delete(k)); }
  S.sets.transfer = meta.items.map((m, i) => ingest(m, arrays, `${i}_`).meta.key);
  showSet('transfer');
  select(S.sets.transfer[0], true);
}

/* ---------------------------------------------------------- body bar */
function bodyBar() {
  const it = act();
  $('#bodyOf').textContent = it ? `· ${title(it)}, the garment follows edits` : '· new samples go on this body';
  $('#betaList')._sync();
  syncSegs();
}
async function pushBody(payload) {
  const it = act();
  const q = { ...payload, key: it?.meta.key };
  const { meta, arrays } = await call('/api/body', q);
  if (meta.item) {
    S.body.beta = Float32Array.from(meta.item.body.beta); S.body.space = meta.item.body.space;
    await updateItem(meta, arrays);
    S.body.verts = arrays.verts;
  } else {
    S.body.beta = Float32Array.from(meta.body.beta); S.body.space = meta.body.space; S.body.verts = arrays.verts;
    showActive(true);
  }
  bodyBar();
}
const setBodyBeta = () => pushBody({ space: S.body.space, beta: Array.from(S.body.beta) });
const randomBody = (space = S.body.space) => pushBody({ randomize: true, space, seed: rseed() });
let bodyTimer = null;
function scheduleBody() { clearTimeout(bodyTimer); bodyTimer = setTimeout(safe(setBodyBeta), 220); }

/* ----------------------------------------------------------- coverage */
function coveredCount() { return S.occ ? S.occ.reduce((a, v) => a + (v > 0.5), 0) : 0; }
function updateCoverageHint() {
  const base = S.occBase, it = act(), hint = $('#occHint');
  const covered = coveredCount();
  let txt = `Painted mask: ${covered} of 6 890 vertices covered.`;
  if (base && it && base.key === it.meta.key) {
    let grown = 0, trimmed = 0;
    for (let i = 0; i < 6890; i++) {
      if (S.occ[i] > 0.5 && base.occ[i] <= 0.5) grown++;
      else if (S.occ[i] <= 0.5 && base.occ[i] > 0.5) trimmed++;
    }
    txt += ` Versus this garment: +${grown} / −${trimmed}.`;
  }
  hint.textContent = txt;
  const on = $('#useOcc').checked;
  $('#occSrcNote').innerHTML = !on ? ''
    : covered ? `The mask below is used. <a href="#" id="gotoOcc">Edit it</a>.`
              : 'No coverage yet. Randomize or paint one.';
  const g = $('#gotoOcc'); if (g) g.onclick = (e) => { e.preventDefault(); setTab('occ'); };
  syncSampleBtn();
}
function syncSampleBtn() {
  const btn = $('#doSample'), on = $('#useOcc').checked, ok = !on || coveredCount() > 0;
  btn.disabled = !ok;
  const label = `${on ? 'Sample with coverage' : 'Sample'} ${S.n}`;
  const span = btn.querySelector('span');
  if (span.textContent !== label) span.textContent = label;
  btn.title = ok ? '' : 'paint or randomize a coverage first';
}
async function occCall(body, label) {
  const { meta, arrays } = await call('/api/occ', body);
  S.occ = arrays.occ;
  applyView(); updateCoverageHint();
  say(`${label} · ${meta.covered} vertices covered`);
  return arrays.occ;
}
async function randomCoverage(cls) {
  if (!cls) throw new Error('pick a class first');
  const occ = await occCall({ random: true, cls }, 'random coverage');
  S.occBase = null;
  return occ;
}
async function occFromGarment(source) {
  const it = act();
  if (!it) throw new Error('pick a garment first');
  if (source !== 'cond' && !meshOK(it)) { say('Meshing the garment…'); await doMesh(it.meta.key); }
  if (source !== 'cond') say('Reading coverage off the mesh…');
  const occ = await occCall({ from_key: it.meta.key, source }, source === 'cond' ? 'coverage from its condition' : 'coverage from garment');
  S.occBase = { key: it.meta.key, occ: occ.slice() };
  updateCoverageHint();
}
async function doEdit() {
  const it = act();
  if (!it) throw new Error('pick a garment first');
  const { meta, arrays } = await timed('Editing the coverage',
    () => call('/api/edit', { key: it.meta.key, occ_w: S.adhere }));
  if (S.paint) setPaint(false);
  const set = S.sets.samples.includes(it.meta.key) ? 'samples' : 'transfer';
  ingest(meta.item, arrays);
  S.sets[set].splice(S.sets[set].indexOf(it.meta.key) + 1, 0, meta.item.key);
  showSet(set);
  select(meta.item.key);
  return meta.item;
}
/* ------------------------------------------------------------ painting */
const ray = new THREE.Raycaster();
const _paintP = new THREE.Vector3();
const pointer = new THREE.Vector2();
let painting = false, pendingIdx = new Set(), pendingVal = 1, spaceHeld = false;
const orbiting = () => S.brush === 2 || spaceHeld;

function paintAt(ev) {
  const r = canvas.getBoundingClientRect();
  pointer.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(pointer, camera);
  const through = $('#brushThrough').checked, mirror = $('#brushMirror').checked && S.mirror;
  const hit = ray.intersectObject(main.body, false)[0];
  if (!hit && !through) return;
  const rad = (+$('#brushR').value) / 100, r2 = rad * rad;
  const pos = main.body.geometry.getAttribute('position'), c = main.body.geometry.getAttribute('color');
  const set = (i) => {
    if ((S.locked && S.locked[i]) || S.occ[i] === pendingVal) return;
    S.occ[i] = pendingVal; pendingIdx.add(i);
    const col = pendingVal > 0.5 ? coverColor : bareColor;
    if (c.itemSize === 4) c.setXYZW(i, col.r, col.g, col.b, 1);
    else c.setXYZ(i, col.r, col.g, col.b);
  };
  for (let i = 0; i < 6890; i++) {
    _paintP.set(pos.getX(i), pos.getY(i), pos.getZ(i));
    if ((through ? ray.ray.distanceSqToPoint(_paintP) : _paintP.distanceToSquared(hit.point)) > r2) continue;
    set(i);
    if (mirror) set(S.mirror[i]);
  }
  c.needsUpdate = true;
}
function setOrbitButtons() {
  const painterly = S.paint && !orbiting();
  controls.mouseButtons = painterly
    ? { LEFT: -1, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE }
    : { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
  canvas.style.cursor = painterly ? 'crosshair' : '';
}
canvas.addEventListener('pointerdown', (ev) => {
  if (!S.paint || ev.button !== 0 || orbiting() || isBusy()) return;
  pendingVal = (ev.shiftKey ? S.brush !== 1 : S.brush === 1) ? 1 : 0;
  painting = true; pendingIdx.clear(); paintAt(ev);
});
canvas.addEventListener('pointermove', (ev) => { if (painting) paintAt(ev); });
window.addEventListener('pointerup', safe(async () => {
  if (!painting) return;
  painting = false;
  if (!pendingIdx.size) return;
  const idx = [...pendingIdx];
  pendingIdx.clear();
  await occCall({ idx, value: pendingVal }, 'painted');
}));
function setPaint(on) {
  S.paint = on;
  for (const id of ['#paintToggle', '#editPaint']) {
    const b = $(id);
    b.classList.toggle('on', on);
    b.textContent = on ? 'Done painting' : 'Paint on body';
  }
  $('#paintTools').hidden = !on;
  $('#paintHint').hidden = !on;
  if (on && !$('#useOcc').checked) { $('#useOcc').checked = true; $('#occOpts').hidden = false; }   // painting implies coverage
  setOrbitButtons(); applyView(); updateCoverageHint(); syncSegs();
}

/* ------------------------------------------------------------- widgets */
function seg(el, items, get, set) {
  el.innerHTML = '';
  items.forEach(([val, label]) => {
    const b = document.createElement('button');
    b.textContent = label; b.dataset.val = val;
    b.onclick = () => { set(val); syncSegs(); };
    el.appendChild(b);
  });
  el._get = get;
}
function syncSegs() {
  $$('.seg').forEach((el) => {
    if (!el._get) return;
    const cur = String(el._get());
    el.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.val === cur));
  });
}
function slider(id, fmt, onChange) {
  const el = $('#' + id), out = el.closest('.slider').querySelector('b');
  const upd = () => { out.textContent = fmt(parseFloat(el.value)); };
  el.oninput = () => { upd(); onChange && onChange(parseFloat(el.value)); };
  upd();
}
function betaSliders(host) {
  host.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    const wrap = document.createElement('div');
    wrap.className = 'slider';
    wrap.innerHTML = `<label>β${i + 1}<b>0.0</b></label><input type="range" min="-4" max="4" step="0.05">`;
    const inp = wrap.querySelector('input'), out = wrap.querySelector('b');
    inp.oninput = () => { out.textContent = (+inp.value).toFixed(1); S.body.beta[i] = +inp.value; scheduleBody(); };
    host.appendChild(wrap);
  }
  host._sync = () => host.querySelectorAll('.slider').forEach((w, i) => {
    w.querySelector('input').value = S.body.beta[i];
    w.querySelector('b').textContent = (+S.body.beta[i]).toFixed(1);
  });
}
function syncDensity() {
  const big = curL() > 4096;
  $('#lvlNote').hidden = !big;
  $('#cfg').closest('.slider').classList.toggle('locked', big);
  $('#samplerHint').textContent = `${S.solver === 'heun' ? 'Heun' : 'Euler'} · cfg ${big ? '1' : (+$('#cfg').value).toFixed(1)}`;
  syncSegs();
}
function setTab(t, keep) {
  S.tab = t;
  $$('#tabs button').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === t));
  $$('.pane').forEach((p) => { p.hidden = p.dataset.pane !== t; });
  if (t !== 'occ' && S.paint) setPaint(false);
  if (t === 'transfer' && S.sets.transfer.length) showSet('transfer');
  if (t === 'sample') { showSet('samples'); if (!keep) clearActive(); }   // Sample starts from an empty body
  applyView();
  targets(); updateCoverageHint();
}
function targets() {
  const it = act();
  const html = it
    ? `<img alt="" src="${it.thumb}"><div><b>${title(it)}</b><span>${subtitle(it)}</span></div>`
    : 'Pick a garment in the preview column';
  for (const id of ['#occTarget', '#tTarget']) {
    $(id).innerHTML = html; $(id).classList.toggle('none', !it);
  }
  for (const id of ['#doEdit', '#doTransfer', '#occFromMesh']) $(id).disabled = !it;
  $('#occFromCond').disabled = !it?.meta.cond;
  syncMesh(); syncSampleBtn();
}
const hasCloud = (it) => !!it && !it.pending && !!it.points && !meshOK(it);   // pointcloud, no usable mesh yet
function meshable() {
  return S.sets[S.set].filter((k) => hasCloud(S.items.get(k)));
}
function syncMesh() {
  const it = act(), ok = hasCloud(it);
  const btn = $('#meshActive');
  btn.disabled = !ok;
  btn.title = ok ? `Mesh ${title(it)} (M)` : meshOK(it) ? 'This garment already has a mesh' : 'Pick a garment first';
  const n = meshable().length;
  $('#meshAllCards').disabled = n === 0;
  $('#meshAllCards').title = n ? `Mesh ${n} card${n === 1 ? '' : 's'} in ${S.set} without a mesh` : 'Nothing to mesh in this set';
}
function applyLook() {
  const th = THEME[S.look.theme];
  document.documentElement.dataset.theme = S.look.theme;
  $('#themeBtn').textContent = S.look.theme === 'light' ? 'Dark' : 'Light';
  bareColor.setHex(th.bare);
  garmentColor.set(S.look.color);
  styleCell(main);
  $('#colorPick').value = S.look.color;
  $('#colorHex').textContent = S.look.color;
  $('#wire').checked = S.look.wire;
  $('#bodyOpacity').value = S.look.bodyOpacity;
  $('#bodyOpacity').closest('.slider').querySelector('b').textContent = `${S.look.bodyOpacity}%`;
  $$('#swatches button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.val === S.look.color));
  if (S.init) { showActive(); rethumbAll(); }
  syncSegs();
}

async function resetAll() {
  const { meta, arrays } = await call('/api/reset', {});
  if (S.paint) setPaint(false);
  S.items.clear(); S.sets = { samples: [], transfer: [] }; S.active = null; S.occBase = null;
  S.body = { space: meta.body.space, beta: Float32Array.from(meta.body.beta), verts: arrays.verts };
  S.occ = arrays.occ;
  $('#useOcc').checked = false; $('#occOpts').hidden = true; $('#paintTools').hidden = true; $('#paintHint').hidden = true;
  setTab('sample', true); showActive(true); renderCards(); updateCoverageHint();
  say('reset · new body');
}

/* ----------------------------------------------------------------- init */
(async function boot() {
  const { meta, arrays } = await call('/api/init');
  S.init = { ...meta, faces: arrays.faces };
  const d = meta.defaults;
  S.cls = d.cls; S.solver = d.solver; S.n = d.n;
  S.body = { space: meta.body.space, beta: Float32Array.from(meta.body.beta), verts: arrays.verts };
  S.occ = arrays.occ;
  S.locked = arrays.locked; S.mirror = arrays.mirror; S.adhere = 1;
  seg($('#adhereSeg'), [[1, 'Normal'], [2, 'Strong']], () => S.adhere, (v) => { S.adhere = +v; });
  $('#level').innerHTML = meta.levels.map((l) => `<option value="${l}"${l === d.L ? ' selected' : ''}>${l.toLocaleString()} points</option>`).join('');
  seg($('#classSeg'), [...meta.classes, meta.any_class].map((c) => [c, c]), () => S.cls, (v) => { S.cls = v; syncSampleBtn(); });
  seg($('#countSeg'), [1, 2, 4, 6, 8].map((n) => [n, String(n)]), () => S.n, (v) => { S.n = +v; syncSampleBtn(); });
  seg($('#tCountSeg'), [2, 4, 6, 8].map((n) => [n, String(n)]), () => S.tn, (v) => { S.tn = +v; $('#doTransfer').firstChild.textContent = `Transfer to ${S.tn} bodies`; });
  seg($('#tSpaceSeg'), [['mixed', 'Mixed'], ...meta.spaces.map((s, i) => [i, s])], () => S.tspace, (v) => { S.tspace = v; });
  seg($('#solverSeg'), [['euler', 'Euler'], ['heun', 'Heun']], solverNow, (v) => { S.solver = v; syncDensity(); });
  seg($('#spaceSeg'), meta.spaces.map((s, i) => [i, s]), () => S.body.space, (v) => safe(randomBody)(+v));
  seg($('#brushSeg'), [[1, 'Add'], [0, 'Erase'], [2, 'Orbit']], () => S.brush, (v) => { S.brush = +v; setOrbitButtons(); });
  seg($('#showSeg'), [['points', 'Points'], ['mesh', 'Mesh'], ['both', 'Both']], () => S.view.show, (v) => { S.view.show = v; applyView(); rethumbAll(); });
  seg($('#colorSeg'), [['plain', 'Flat'], ['depth', 'Depth']], () => S.view.color, (v) => { S.view.color = v; showActive(); rethumbAll(); });
  seg($('#finishSeg'), [['matte', 'Matte'], ['satin', 'Satin'], ['glossy', 'Glossy'], ['metal', 'Metal']],
      () => S.look.finish, (v) => { S.look.finish = v; applyLook(); });
  seg($('#setSeg'), [['samples', 'Samples'], ['transfer', 'Transfer']], () => S.set, (v) => showSet(v));
  $('#doTransfer').firstChild.textContent = `Transfer to ${S.tn} bodies`;

  $('#swatches').innerHTML = SWATCHES.map((c) => `<button data-val="${c}" style="background:${c}" title="${c}"></button>`).join('');
  $$('#swatches button').forEach((b) => { b.onclick = () => { S.look.color = b.dataset.val; applyLook(); }; });
  $('#colorPick').onchange = () => { S.look.color = $('#colorPick').value; applyLook(); };
  $('#wire').onchange = () => { S.look.wire = $('#wire').checked; applyLook(); };
  $('#lookSave').onclick = () => {
    try { localStorage.setItem(LOOK_KEY, JSON.stringify(S.look)); say('appearance saved as default'); }
    catch { say('⚠ could not save (private browsing?)'); }
  };
  $('#lookReset').onclick = () => {
    S.look = { ...LOOK_DEFAULT };
    try { localStorage.removeItem(LOOK_KEY); } catch { /* ignore */ }
    applyLook(); say('appearance restored');
  };
  $('#lookBtn').onclick = () => { $('#lookPop').hidden = !$('#lookPop').hidden; $('#lookBtn').classList.toggle('on', !$('#lookPop').hidden); };
  $('#themeBtn').onclick = () => { S.look.theme = S.look.theme === 'light' ? 'dark' : 'light'; applyLook(); };

  slider('steps', (v) => v.toFixed(0));
  slider('cfg', (v) => v.toFixed(1), syncDensity);
  slider('brushR', (v) => `${v} cm`);
  let opTimer = null;
  slider('bodyOpacity', (v) => `${v}%`, (v) => {
    S.look.bodyOpacity = v; styleCell(main); applyView();
    clearTimeout(opTimer); opTimer = setTimeout(rethumbAll, 250);
  });
  betaSliders($('#betaList'));

  $$('#tabs button').forEach((b) => { b.onclick = () => setTab(b.dataset.tab); });
  $('#level').onchange = syncDensity;
  $('#useOcc').onchange = safe(async () => {
    const on = $('#useOcc').checked;
    $('#occOpts').hidden = !on;
    if (on) {
      if (!S.view.body) { S.view.body = true; $('#toggleBody').classList.add('on'); }   // the patch must be visible
      if (coveredCount() === 0) await randomCoverage(act()?.meta.cls || S.cls);         // always start from a mask
    }
    updateCoverageHint(); applyView();
  });
  $('#randBody').onclick = safe(() => randomBody());
  $('#resetBody').onclick = safe(() => { S.body.beta = new Float32Array(10); return setBodyBeta(); });
  $('#seedLock').onclick = () => {
    S.seedLock = !S.seedLock;
    $('#seedLock').classList.toggle('on', S.seedLock);
    $('#shackle').setAttribute('d', S.seedLock ? 'M8 11V7a4 4 0 0 1 8 0' : 'M8 11V7a4 4 0 0 1 7-2.5');
    $('#seedNote').innerHTML = S.seedLock ? 'Seed locked, Sample redraws this set (to ~1 to 2 mm; bf16).'
                                          : 'Sample <i>k</i> uses seed + <i>k</i>. Lock the seed to redraw the same set.';
  };
  $('#doSample').onclick = safe(doSample);
  $('#meshActive').onclick = safe(() => timed('Meshing', async () => {
    await doMesh();
    if (S.view.show === 'points') { S.view.show = 'mesh'; applyView(); syncSegs(); rethumbAll(); }
  }));
  $('#meshAllCards').onclick = safe(doMeshAll);
  $('#doTransfer').onclick = safe(doTransfer);
  $('#doEdit').onclick = safe(() => doEdit());
  $('#doOccSample').onclick = safe(() => {
    $('#useOcc').checked = true; $('#occOpts').hidden = false;
    const it = act(); if (it) S.cls = it.meta.cls;
    updateCoverageHint(); setTab('sample', true);
    return doSample();
  });
  const paintToggle = safe(async () => {
    const it = act();
    // painting a garment's coverage reads it off that garment first
    if (!S.paint && it && S.occBase?.key !== it.meta.key) await occFromGarment(it.meta.cond ? 'cond' : 'mesh');
    else if (!S.paint && !it && coveredCount() === 0) await randomCoverage(S.cls);
    setPaint(!S.paint);
  });
  $('#paintToggle').onclick = paintToggle;
  $('#editPaint').onclick = paintToggle;
  $('#occFromMesh').onclick = safe(() => occFromGarment('mesh'));
  $('#occFromCond').onclick = safe(() => occFromGarment('cond'));
  $('#occRandom').onclick = safe(() => randomCoverage(act()?.meta.cls || S.cls));
  $('#occRandomEdit').onclick = safe(() => randomCoverage(act()?.meta.cls || S.cls));
  $('#occClear').onclick = safe(() => occCall({ clear: true }, 'cleared'));
  $('#clearSet').onclick = safe(() => forget(S.sets[S.set].slice()));
  $('#toggleBody').onclick = () => {
    S.view.body = !S.view.body;
    $('#toggleBody').classList.toggle('on', S.view.body); applyView();
  };
  $('#toggleBody').classList.add('on');
  $('#physBtn').onclick = () => {
    S.phys.on = !S.phys.on;
    if (S.phys.on) physArm(act());
    else { physReset(); setMesh(main, meshOK(act()) ? act().mverts : null, act()?.mfaces); }
    applyView();
  };
  $('#frameBtn').onclick = () => frameView();
  $('#resetAll').onclick = safe(resetAll);
  busyCancel.onclick = () => cancelFlow();

  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.code === 'Space' && S.paint) { e.preventDefault(); if (!spaceHeld) { spaceHeld = true; setOrbitButtons(); } return; }
    if (e.key === 'Escape') { if (S.paint) setPaint(false); $('#lookPop').hidden = true; $('#lookBtn').classList.remove('on'); return; }
    if (isBusy()) return;                                   // no shortcuts while a flow is running
    const k = e.key.toLowerCase();
    if (k === 's' && !$('#doSample').disabled) safe(doSample)();
    if (k === 't' && act()) safe(doTransfer)();
    if (k === 'm' && !$('#meshActive').disabled) $('#meshActive').click();
    if (k === 'b') safe(randomBody)();
    if (k === 'f') frameView();
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); step(1); }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); step(-1); }
  });
  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && spaceHeld) { spaceHeld = false; setOrbitButtons(); }
  });

  styleCell(thumb);
  applyLook(); syncDensity(); setTab('sample'); renderCards(); updateCoverageHint();
  layout(); showActive(true); draw();
  say('ready, sample a garment');
})();
