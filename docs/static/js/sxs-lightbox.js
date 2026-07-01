/* ============================================================
   sxs-lightbox.js — click a Side-by-Side Render video to open it large.

   Scope: only videos inside [data-sxs-zoom] figures (§02 "Side-by-Side Renders").
   In the lightbox: click toggles play/pause; mouse wheel zooms (centred on the
   cursor) so a paused frame can be inspected up close; drag pans when zoomed;
   double-click resets; Esc / backdrop / × closes.
   ============================================================ */
(function () {
  function build() {
    const lb = document.createElement('div');
    lb.className = 'sxs-lb';
    lb.hidden = true;
    lb.innerHTML = [
      '<div class="sxs-lb__backdrop" data-close></div>',
      '<div class="sxs-lb__stage" data-stage>',
      '  <video class="sxs-lb__video" data-video playsinline loop muted></video>',
      '</div>',
      '<button class="sxs-lb__close" data-close type="button" aria-label="Close">&times;</button>',
      '<div class="sxs-lb__hint">click: play / pause &middot; scroll: zoom &middot; drag: pan &middot; double-click: reset &middot; esc: close</div>',
    ].join('');
    document.body.appendChild(lb);
    return lb;
  }

  const lb = build();
  const stage = lb.querySelector('[data-stage]');
  const video = lb.querySelector('[data-video]');

  let s = 1, tx = 0, ty = 0;          // zoom scale + pan translate (px)
  let down = null, moved = false;

  function apply() {
    video.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + s + ')';
    stage.classList.toggle('is-zoomed', s > 1.01);
  }
  function reset() { s = 1; tx = 0; ty = 0; apply(); }

  function open(src) {
    if (!src) return;
    video.src = src;
    lb.hidden = false;
    document.body.classList.add('sxs-lb-open');
    reset();
    video.currentTime = 0;
    video.play().catch(function () {});
  }
  function close() {
    lb.hidden = true;
    document.body.classList.remove('sxs-lb-open');
    video.pause();
    video.removeAttribute('src');
    video.load();
  }

  // open triggers — every video inside a [data-sxs-zoom] figure
  document.querySelectorAll('[data-sxs-zoom] video').forEach(function (v) {
    const cell = v.closest('.trajectory-cell') || v;
    cell.classList.add('sxs-zoomable');
    cell.addEventListener('click', function (e) {
      e.preventDefault();
      const src = v.currentSrc || (v.querySelector('source') && v.querySelector('source').getAttribute('src'));
      open(src);
    });
  });

  // close
  lb.querySelectorAll('[data-close]').forEach(function (el) {
    el.addEventListener('click', close);
  });
  document.addEventListener('keydown', function (e) {
    if (!lb.hidden && e.key === 'Escape') close();
  });

  // wheel zoom, centred on the cursor
  stage.addEventListener('wheel', function (e) {
    e.preventDefault();
    const rect = video.getBoundingClientRect();
    const cx = e.clientX - (rect.left + rect.width / 2);
    const cy = e.clientY - (rect.top + rect.height / 2);
    const px = (cx - tx) / s, py = (cy - ty) / s;     // point under cursor (pre-transform)
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    s = Math.min(8, Math.max(1, s * factor));
    if (s <= 1.0001) { tx = 0; ty = 0; }
    else { tx = cx - px * s; ty = cy - py * s; }       // keep that point under the cursor
    apply();
  }, { passive: false });

  // drag to pan (when zoomed) + click (no drag) toggles play/pause
  stage.addEventListener('pointerdown', function (e) {
    down = { x: e.clientX, y: e.clientY, tx: tx, ty: ty };
    moved = false;
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', function (e) {
    if (!down) return;
    const dx = e.clientX - down.x, dy = e.clientY - down.y;
    if (Math.abs(dx) + Math.abs(dy) > 5) moved = true;
    if (s > 1.01) {
      tx = down.tx + dx; ty = down.ty + dy;
      stage.classList.add('is-panning');
      apply();
    }
  });
  stage.addEventListener('pointerup', function () {
    stage.classList.remove('is-panning');
    if (down && !moved) {
      if (video.paused) video.play().catch(function () {}); else video.pause();
    }
    down = null;
  });
  stage.addEventListener('dblclick', function (e) { e.preventDefault(); reset(); });
})();
