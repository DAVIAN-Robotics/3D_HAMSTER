/* ============================================================
   synced-pcd.js — temporal point-cloud player locked to a <video>.

   The .ply frames were sampled from the same camera stream as the rollout
   mp4, so we drive the displayed frame from the video's normalized time:
       idx = floor(video.currentTime / video.duration * n_frames)
   => the 3D playback stays in exact sync with the mp4, and the camera is
   freely orbitable (OrbitControls). One persistent viewer; setSource swaps
   episodes.

   Data (see tools/make_pcd_frames.py):
     frames.json : { n_frames, n_points, center, max_dim, traj{points,colors} }
     frames.bin  : [positions f32 (N*P*3)][colors u8 (N*P*3)]
   ============================================================ */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const BG = 0xf7f8fa;
const POINT_PX = 2.6;
const TRAJ_PX = 3.0;

export function createSyncedPcd(host, video) {
  let renderer, scene, camera, controls, io, ro;
  let scenePts = null, trajPts = null;
  let positions = null, colors = null, N = 0, P = 0;
  let curIdx = -1, rafId = null, visible = true, token = 0;

  function ensure() {
    if (renderer) return;
    scene = new THREE.Scene();
    scene.background = new THREE.Color(BG);
    const w = host.clientWidth || 480, h = host.clientHeight || 360;
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h, false);
    // show point colours as raw RGB (skip the linear->sRGB encode that otherwise
    // brightens/washes the cloud) — matches the Open3D overview clips.
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    renderer.domElement.className = 'synced-pcd__gl';
    host.appendChild(renderer.domElement);

    camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 1000);
    camera.up.set(0, -1, 0);              // camera frame: image-y points down

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.rotateSpeed = 0.6;

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

  function disposePoints(p) {
    if (!p) return;
    scene.remove(p);
    p.geometry.dispose();
    p.material.dispose();
  }

  function build() {
    disposePoints(scenePts);
    disposePoints(trajPts);

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(P * 3), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(P * 3), 3, true));
    const m = new THREE.PointsMaterial({ size: POINT_PX, sizeAttenuation: false, vertexColors: true });
    scenePts = new THREE.Points(g, m);
    scenePts.frustumCulled = false;
    scene.add(scenePts);
  }

  function buildTraj(traj) {
    if (!traj || !traj.points || !traj.points.length) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(traj.points), 3));
    g.setAttribute('color', new THREE.BufferAttribute(Uint8Array.from(traj.colors), 3, true));
    const m = new THREE.PointsMaterial({ size: TRAJ_PX, sizeAttenuation: false, vertexColors: true });
    trajPts = new THREE.Points(g, m);
    trajPts.frustumCulled = false;
    scene.add(trajPts);
  }

  function setFrame(idx, force) {
    if (idx === curIdx && !force) return;
    if (idx < 0) idx = 0; else if (idx >= N) idx = N - 1;
    curIdx = idx;
    const pa = scenePts.geometry.getAttribute('position');
    pa.array.set(positions.subarray(idx * P * 3, (idx + 1) * P * 3));
    pa.needsUpdate = true;
    const ca = scenePts.geometry.getAttribute('color');
    ca.array.set(colors.subarray(idx * P * 3, (idx + 1) * P * 3));
    ca.needsUpdate = true;
  }

  function frameCamera(center, maxDim) {
    const c = new THREE.Vector3(center[0], center[1], center[2]);
    // frontal view ~ the real RGB camera (looking along +z, up = -y).
    // 0.85 => ~20% closer default framing. +y is "down" in camera frame, so a
    // positive y offset starts the camera below the scene, looking up (low angle).
    camera.position.set(c.x, c.y + maxDim * 0.1, c.z - maxDim * 0.85);
    camera.lookAt(c);
    controls.target.copy(c);
    controls.minDistance = maxDim * 0.05;
    controls.maxDistance = maxDim * 8;
    controls.update();
  }

  function loop() {
    rafId = requestAnimationFrame(loop);
    if (!renderer || !visible) return;
    if (scenePts && N && video && video.duration) {
      const idx = Math.floor((video.currentTime / video.duration) * N);
      setFrame(idx);
    }
    controls.update();
    renderer.render(scene, camera);
  }

  async function setSource(jsonUrl) {
    const my = ++token;
    const base = jsonUrl.replace(/frames\.json$/, '');
    const meta = await fetch(jsonUrl).then((r) => r.json());
    const buf = await fetch(base + 'frames.bin').then((r) => r.arrayBuffer());
    if (my !== token) return;
    N = meta.n_frames; P = meta.n_points;
    positions = new Float32Array(buf, 0, N * P * 3);
    colors = new Uint8Array(buf, N * P * 3 * 4, N * P * 3);
    ensure();
    build();
    buildTraj(meta.traj);
    curIdx = -1;
    setFrame(0, true);
    frameCamera(meta.center, meta.max_dim);
  }

  return { setSource };
}

// expose for the (non-module) gallery script; the module finishes executing
// well before the gallery scrolls into view, so this is ready by activation.
window.createSyncedPcd = createSyncedPcd;
