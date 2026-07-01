/* ============================================================
   static-pcd.js — static PLY viewer + carousel (§02).

   A direct port of the §03 Real-Robot interactive viewer (synced-pcd.js): the
   renderer / camera / OrbitControls / continuous render-loop are IDENTICAL, so
   drag-orbit, scroll-zoom and pan feel exactly the same. The only difference is
   the data source — §02 shows a single static comparison_3d.ply (scene point
   cloud + baked method trajectories) instead of §03's video-synced temporal
   frame stream, so there is no <video> and no per-frame swap.

   Each viewer is wired into a prev/next/dots/keyboard carousel of scenes.
   ============================================================ */
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const BG = 0xf7f8fa;
const POINT_PX = 2.6;          // == §03 scene point size
const MAX_POINTS = 60000;
const DROID_Z_MAX = 2.0;       // DROID clouds: drop the far back wall for framing
// The comparison_3d.ply files are now baked OURS-ONLY: the RoboBrain (orange/red)
// and Gemini (green) tubes are stripped offline and our trajectory is recoloured
// to a red->blue jet gradient. So the in-browser hue filter must be OFF — a
// gradient tube is no longer blue-dominant and dropMarker() would shred it.
const OURS_ONLY = false;

/* ---------- deterministic subsample ---------- */
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- PLY -> subsampled {positions, colors, center, maxDim} ---------- */
const CACHE = new Map();
const INFLIGHT = new Map();
let _loader = null;
const getLoader = () => (_loader || (_loader = new PLYLoader()));

function process(geometry, dataset) {
  const posAttr = geometry.getAttribute('position');
  const colAttr = geometry.getAttribute('color');
  const n = posAttr.count;
  const src = posAttr.array;

  // build kept indices: optional DROID z-crop + (OURS_ONLY) keep only our BLUE
  // trajectory. The baked method tubes are highly saturated gradient colours
  // (one channel ~0, another high) — ours = blue, RoboBrain = orange/red,
  // Gemini = green; natural scene colours never have a near-zero channel. So a
  // "marker" = very saturated (min < 0.15 && max > 0.45 in sRGB); keep markers
  // where blue dominates (ours), drop the rest. PLYLoader linearises ply colours,
  // so convert back to sRGB before testing hue.
  const cs0 = colAttr ? colAttr.array : null;
  const cf = cs0 instanceof Float32Array ? 1 : 1 / 255;
  function l2s(c) {
    c = c < 0 ? 0 : c;
    return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  }
  function dropMarker(i) {
    if (!cs0) return false;
    const r = l2s(cs0[i * 3] * cf), g = l2s(cs0[i * 3 + 1] * cf), b = l2s(cs0[i * 3 + 2] * cf);
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const isMarker = mn < 0.15 && mx > 0.45;     // a baked, very-saturated tube
    const bDom = b >= r && b >= g;               // blue-dominant == our 3D HAMSTER
    return isMarker && !bDom;
  }
  let idx = [];
  for (let i = 0; i < n; i++) {
    if (dataset === 'droid' && src[i * 3 + 2] > DROID_Z_MAX) continue;
    if (OURS_ONLY && dropMarker(i)) continue;
    idx.push(i);
  }

  // subsample (Fisher-Yates prefix with a fixed seed -> stable across reloads)
  const rand = mulberry32(0xC0FFEE);
  if (idx.length > MAX_POINTS) {
    for (let i = 0; i < MAX_POINTS; i++) {
      const j = i + Math.floor(rand() * (idx.length - i));
      const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
    idx = idx.slice(0, MAX_POINTS);
  }

  const k = idx.length;
  const positions = new Float32Array(k * 3);
  const colors = new Uint8Array(k * 3);
  const cs = colAttr ? colAttr.array : null;
  const cScale = cs && !(cs instanceof Float32Array) ? 1 : 255; // f32 0..1 -> 0..255
  let xmin = Infinity, ymin = Infinity, zmin = Infinity;
  let xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
  for (let i = 0; i < k; i++) {
    const s = idx[i] * 3;
    const x = src[s], y = src[s + 1], z = src[s + 2];
    positions[i * 3] = x; positions[i * 3 + 1] = y; positions[i * 3 + 2] = z;
    if (cs) {
      colors[i * 3]     = cScale === 255 ? cs[s] * 255 : cs[s];
      colors[i * 3 + 1] = cScale === 255 ? cs[s + 1] * 255 : cs[s + 1];
      colors[i * 3 + 2] = cScale === 255 ? cs[s + 2] * 255 : cs[s + 2];
    } else { colors[i * 3] = colors[i * 3 + 1] = colors[i * 3 + 2] = 150; }
    if (x < xmin) xmin = x; if (x > xmax) xmax = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    if (z < zmin) zmin = z; if (z > zmax) zmax = z;
  }
  const center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2];
  const maxDim = Math.max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-3);
  return { positions, colors, center, maxDim };
}

function loadScene(url, dataset) {
  const key = `${dataset || ''}|${url}`;
  if (CACHE.has(key)) return Promise.resolve(CACHE.get(key));
  if (INFLIGHT.has(key)) return INFLIGHT.get(key);
  const p = new Promise((res, rej) => {
    getLoader().load(url, (g) => { try { res(process(g, dataset)); g.dispose && g.dispose(); } catch (e) { rej(e); } }, undefined, rej);
  }).then((d) => { CACHE.set(key, d); INFLIGHT.delete(key); return d; })
    .catch((e) => { INFLIGHT.delete(key); throw e; });
  INFLIGHT.set(key, p);
  return p;
}
function prefetch(url, dataset) { if (url) loadScene(url, dataset).catch(() => {}); }

/* ---------- viewer (renderer/camera/controls/loop == §03 synced-pcd) ---------- */
function createStaticPcd(host) {
  let renderer, scene, camera, controls, io, ro;
  let pts = null, visible = true, rafId = null, token = 0;

  function ensure() {
    if (renderer) return;
    scene = new THREE.Scene();
    scene.background = new THREE.Color(BG);
    const w = host.clientWidth || 480, h = host.clientHeight || 360;
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h, false);
    // keep three.js' default sRGB output encode — the brighter, punchier colour
    // the §02 comparison clouds were tuned for (NOT the §03 raw-RGB look).
    renderer.domElement.className = 'pcd-host__gl';
    host.appendChild(renderer.domElement);

    camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 1000);
    camera.up.set(0, -1, 0);              // camera-frame clouds: image-y points down

    controls = makeControls();

    ro = new ResizeObserver(() => {
      const w2 = host.clientWidth, h2 = host.clientHeight;
      if (w2 && h2) {
        renderer.setSize(w2, h2, false);
        camera.aspect = w2 / h2;
        camera.updateProjectionMatrix();
      }
    });
    ro.observe(host);

    io = new IntersectionObserver((es) => {
      for (const e of es) visible = e.isIntersecting;
    }, { rootMargin: '100px' });
    io.observe(host);

    loop();
  }

  // OrbitControls captures camera.up ONCE (it builds the orbit-axis quaternion at
  // construction, r160). So whenever a slide needs a different up-axis we must
  // rebuild the controls — otherwise the orbit would tumble around the old axis.
  function makeControls() {
    const c = new OrbitControls(camera, renderer.domElement);
    c.enableDamping = true;
    c.dampingFactor = 0.1;
    c.rotateSpeed = 0.6;
    return c;
  }

  function disposePoints(p) {
    if (!p) return;
    scene.remove(p);
    p.geometry.dispose();
    p.material.dispose();
  }

  function build(d) {
    disposePoints(pts);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(d.positions, 3));
    g.setAttribute('color', new THREE.BufferAttribute(d.colors, 3, true));
    const m = new THREE.PointsMaterial({ size: POINT_PX, sizeAttenuation: false, vertexColors: true });
    pts = new THREE.Points(g, m);
    pts.frustumCulled = false;
    scene.add(pts);
  }

  function frameCamera(center, maxDim, pov, up) {
    const c = new THREE.Vector3(center[0], center[1], center[2]);

    // up-axis: default (0,-1,0) for the camera-frame clouds; a slide may override
    // it (e.g. the world-frame banana scene needs (0,0,1)). Rebuild controls when
    // it actually changes so the orbit axis follows.
    const ux = up ? up[0] : 0, uy = up ? up[1] : -1, uz = up ? up[2] : 0;
    if (camera.up.x !== ux || camera.up.y !== uy || camera.up.z !== uz) {
      camera.up.set(ux, uy, uz);
      controls.dispose();
      controls = makeControls();
    }

    if (pov && pov.length === 3) {
      // per-slide camera offset (fraction of maxDim) for scenes whose default
      // frontal frame comes out tilted.
      camera.position.set(c.x + maxDim * pov[0], c.y + maxDim * pov[1], c.z + maxDim * pov[2]);
    } else {
      // frontal view ~ the real RGB camera (look along +z, up = -y). 1.2 == the
      // original §02 framing distance (less close-up than §03's 0.85); +y is
      // "down", so a small positive y offset starts the camera slightly below.
      camera.position.set(c.x, c.y + maxDim * 0.1, c.z - maxDim * 1.2);
    }
    camera.lookAt(c);
    controls.target.copy(c);
    controls.minDistance = maxDim * 0.05;
    controls.maxDistance = maxDim * 8;
    controls.update();
  }

  function loop() {
    rafId = requestAnimationFrame(loop);
    if (!renderer || !visible) return;
    controls.update();
    renderer.render(scene, camera);
  }

  async function setSource({ ply, dataset, pov, up } = {}) {
    const my = ++token;
    try {
      const d = await loadScene(ply, dataset);
      if (my !== token) return;
      ensure();
      build(d);
      frameCamera(d.center, d.maxDim, pov, up);
    } catch (e) { console.error('[static-pcd]', e); }
  }

  return { setSource };
}

/* ---------- carousel ---------- */
let _lastFocused = null;

function initCarousel(root) {
  const hostEl = root.querySelector('[data-viewer]');
  const slidesEl = root.querySelector('script[type="application/json"][data-slides]');
  if (!hostEl || !slidesEl) return;
  let slides;
  try { slides = JSON.parse(slidesEl.textContent); } catch (e) { return; }
  if (!Array.isArray(slides) || !slides.length) return;

  const viewer = createStaticPcd(hostEl);
  const prevBtn = root.querySelector('[data-prev]');
  const nextBtn = root.querySelector('[data-next]');
  const cap = root.querySelector('[data-cap]');
  const dots = root.querySelector('[data-dots]');
  const cur = root.querySelector('[data-counter-current]');
  const tot = root.querySelector('[data-counter-total]');
  let idx = 0;

  if (dots) slides.forEach((_, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'pcd-carousel__dot';
    b.setAttribute('role', 'tab'); b.setAttribute('aria-label', `Scene ${i + 1}`);
    b.addEventListener('click', () => go(i));
    dots.appendChild(b);
  });

  function render() {
    const s = slides[idx];
    if (cap) cap.textContent = s.caption || '';
    if (cur) cur.textContent = String(idx + 1);
    if (tot) tot.textContent = String(slides.length);
    if (dots) Array.from(dots.children).forEach((d, i) => {
      d.classList.toggle('is-active', i === idx);
      d.setAttribute('aria-selected', i === idx ? 'true' : 'false');
    });
  }

  function show() {
    const s = slides[idx];
    viewer.setSource({ ply: s.ply, dataset: s.dataset, pov: s.pov || null, up: s.up || null });
    render();
    const nx = slides[(idx + 1) % slides.length], pv = slides[(idx - 1 + slides.length) % slides.length];
    prefetch(nx.ply, nx.dataset); prefetch(pv.ply, pv.dataset);
  }
  function go(n) { idx = (n + slides.length) % slides.length; show(); }

  prevBtn && prevBtn.addEventListener('click', () => go(idx - 1));
  nextBtn && nextBtn.addEventListener('click', () => go(idx + 1));
  root.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); go(idx - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); go(idx + 1); }
  });
  root.addEventListener('focusin', () => { _lastFocused = root; root.classList.add('is-focused'); });
  root.addEventListener('focusout', () => root.classList.remove('is-focused'));

  show();
}

document.addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  const ae = document.activeElement;
  if (ae && ae.closest && ae.closest('.pcd-carousel')) return;
  if (!_lastFocused || !_lastFocused.isConnected) return;
  if (ae && /^(INPUT|TEXTAREA|SELECT)$/i.test(ae.tagName)) return;
  const btn = _lastFocused.querySelector(e.key === 'ArrowLeft' ? '[data-prev]' : '[data-next]');
  if (btn) { e.preventDefault(); btn.click(); }
});

function initAll() { document.querySelectorAll('.pcd-carousel').forEach(initCarousel); }
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initAll);
else initAll();
