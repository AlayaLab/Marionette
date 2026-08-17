/* Coded animations for the System section — one per pipeline stage.
   Nothing here is video. Each stage is drawn live in a canvas from real exported
   arrays (static/data/actions.json, static/data/pose.json):

     0  ActionGPT    a waterfall of the actual per-frame action-id stream
     1  PoseGPT      a real generated episode, orbited in 3D
     2  Bridge       the same episode projected + rasterized, live, in RGB channels
     3  Observation  the chunk-relay schedule that carries appearance

   Colours match the demo film and the rest of the page. Game-native bank/motion
   numbers are deliberately not shown; only the contiguous ids the model predicts.

   Only the selected stage animates. Each selection restarts its clock, runs once
   through, and eases to a stop instead of looping. */
(function () {
  'use strict';

  var C = {
    neural: '#8a93ff', harness: '#59e6be', gold: '#e7c36a', red: '#f26d5b',
    ink: '#ece7db', mute: '#9a978d', faint: '#6a6862', bg: '#0a0a0d'
  };
  var MONO = '11px ui-monospace, "JetBrains Mono", Menlo, monospace';
  var MONO_S = '9.5px ui-monospace, "JetBrains Mono", Menlo, monospace';
  var DISP = '600 15px "Space Grotesk", sans-serif';

  var DATA = { actions: null, pose: null };
  var RUN_SECS = 12;      // each stage plays this long, then settles
  var EASE_SECS = 1.6;

  function lerp(a, b, p) { return a + (b - a) * p; }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function easeInOut(p) { return p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2; }

  function roundRect(g, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    g.beginPath();
    g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r);
    g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r);
    g.arcTo(x, y, x + w, y, r);
    g.closePath();
  }

  /* ---------------- 3D helpers (hand-rolled; no libraries) ---------------- */
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function norm(a) {
    var n = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / n, a[1] / n, a[2] / n];
  }

  function makeCam(az, el, dist, target) {
    var ce = Math.cos(el), se = Math.sin(el);
    var eye = [target[0] + dist * ce * Math.sin(az),
               target[1] + dist * se,
               target[2] + dist * ce * Math.cos(az)];
    var f = norm(sub(target, eye));
    var r = norm(cross(f, [0, 1, 0]));
    var u = cross(r, f);
    return { eye: eye, f: f, r: r, u: u };
  }

  /* The film's camera (demo_video/pipe_act2_render.py): the eye orbits a SMOOTHED pair
     centre at a fixed elevation ratio, with the radius derived from how far the joints
     actually spread. Both the centre path and the radius are precomputed in export_data.py
     so this stays cheap and stays identical to the film. */
  function filmCam(P, t) {
    var c = P.cam, n = c.ctr.length;
    var fi = clamp(t * P.fps, 0, n - 1.001);
    var i0 = Math.floor(fi), a = fi - i0;
    var ctr = [0, 1, 2].map(function (k) { return lerp(c.ctr[i0][k], c.ctr[i0 + 1][k], a) / 100; });
    var r = c.radius_cm / 100;
    var ang = (c.az0_deg + c.az_per_s * t) * Math.PI / 180;
    var eye = [ctr[0] + Math.sin(ang) * r, ctr[1] + c.elev_ratio * r, ctr[2] + Math.cos(ang) * r];
    var target = [ctr[0], ctr[1] + c.target_up_cm / 100, ctr[2]];
    var f = norm(sub(target, eye));
    var rr = norm(cross(f, [0, 1, 0]));
    return { eye: eye, f: f, r: rr, u: cross(rr, f), fov: c.fov_deg * Math.PI / 180 };
  }

  // blend the two nearest keyframes so 15 fps data plays smoothly off the wall clock
  function jointsAt(ent, t, fps) {
    var n = ent.joints.length;
    var fi = clamp(t * fps, 0, n - 1.001);
    var i0 = Math.floor(fi), a = fi - i0;
    var A = ent.joints[i0], B = ent.joints[i0 + 1], out = new Array(A.length);
    for (var k = 0; k < A.length; k++) {
      out[k] = [lerp(A[k][0], B[k][0], a) / 100, lerp(A[k][1], B[k][1], a) / 100,
                lerp(A[k][2], B[k][2], a) / 100];
    }
    return out;
  }

  function projF(p, cam, w, h) {
    var d = sub(p, cam.eye), z = dot(d, cam.f);
    if (z <= 0.05) return null;
    var fpx = (0.5 * h) / Math.tan(cam.fov / 2);
    return [w / 2 + fpx * dot(d, cam.r) / z, h / 2 - fpx * dot(d, cam.u) / z, z];
  }

  // fade a colour toward the page ink, as the film's fade() does
  function fadeCol(rgb, a) {
    a = clamp(a, 0, 1);
    return 'rgb(' + [0, 1, 2].map(function (i) {
      return Math.round(10 + (rgb[i] - 10) * a);
    }).join(',') + ')';
  }
  var RGB = { red: [242, 109, 91], neural: [138, 147, 255], gold: [231, 195, 106],
              harness: [89, 230, 190] };

  // returns [sx, sy, depth]; depth <= 0 means behind the camera
  function project(p, cam, w, h, fov) {
    var d = sub(p, cam.eye);
    var z = dot(d, cam.f);
    if (z <= 0.05) return [0, 0, z];
    var fx = (w / 2) / Math.tan((fov || 0.62) / 2);
    return [w / 2 + fx * dot(d, cam.r) / z, h / 2 - fx * dot(d, cam.u) / z, z];
  }

  function frameOf(ent, i) {
    var js = ent.joints[i];
    return js.map(function (p) { return [p[0] / 100, p[1] / 100, p[2] / 100]; });
  }

  /* ================= stage 0 — action-id waterfall ================= */
  // Runs of one id collapse into a block, so the eye reads "how long each
  // decision was held" rather than a flicker of identical numbers.
  function runsOf(seq) {
    var out = [], i = 0;
    while (i < seq.length) {
      var j = i;
      while (j + 1 < seq.length && seq[j + 1] === seq[i]) j++;
      out.push({ id: seq[i], a: i, b: j + 1 });
      i = j + 1;
    }
    return out;
  }
  var RUNS = null;

  function drawActions(g, w, h, t, settle) {
    var A = DATA.actions;
    if (!A) return;
    if (!RUNS) RUNS = { m: runsOf(A.monster), n: runsOf(A.hunter) };

    var PPF = 3.1;                       // pixels per source frame
    var NOW = h * 0.30;                  // the "current frame" line
    var speed = A.fps * PPF * settle;    // px per second, eased to 0 at the end
    var scroll = t * speed;

    var lanes = [
      { key: 'm', label: 'monster', col: C.gold, x: w * 0.09, vocab: A.vocab.monster },
      { key: 'n', label: 'hunter', col: C.neural, x: w * 0.56, vocab: A.vocab.hunter }
    ];
    var LW = w * 0.35;

    lanes.forEach(function (L) {
      g.save();
      // lane header
      g.font = MONO;
      g.fillStyle = C.faint;
      g.textAlign = 'left';
      g.fillText(L.label.toUpperCase() + '  ·  vocabulary ' + L.vocab, L.x, 18);

      // lane body, clipped
      g.beginPath();
      g.rect(L.x, 28, LW, h - 34);
      g.clip();

      RUNS[L.key].forEach(function (r) {
        // frame index -> y, with time scrolling upward past the NOW line
        var y0 = NOW + r.a * PPF - scroll;
        var y1 = NOW + r.b * PPF - scroll;
        if (y1 < 24 || y0 > h) return;
        var live = y0 <= NOW && y1 > NOW;
        var past = y1 <= NOW;
        var hh = Math.max(2, y1 - y0 - 1.5);
        g.globalAlpha = past ? 0.20 : (live ? 1 : 0.52);
        g.fillStyle = live ? L.col : 'rgba(236,231,219,0.13)';
        roundRect(g, L.x, y0, LW, hh, 3);
        g.fill();
        if (live) {
          g.globalAlpha = 0.25;
          g.strokeStyle = L.col;
          g.lineWidth = 1;
          g.stroke();
        }
        if (hh > 11) {
          g.globalAlpha = past ? 0.45 : 1;
          g.fillStyle = live ? C.bg : C.mute;
          g.font = live ? DISP : MONO_S;
          g.textAlign = 'left';
          g.fillText('id ' + r.id, L.x + 9, y0 + Math.min(hh - 4, 13));
          if (hh > 26) {
            g.font = MONO_S;
            g.globalAlpha = live ? 0.75 : 0.5;
            g.fillText(((r.b - r.a) / A.fps).toFixed(2) + ' s held', L.x + 9, y0 + 27);
          }
        }
      });
      g.restore();
    });

    // the now line
    g.strokeStyle = 'rgba(236,231,219,0.35)';
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, NOW);
    g.lineTo(w, NOW);
    g.stroke();
    g.fillStyle = C.mute;
    g.font = MONO_S;
    g.textAlign = 'right';
    g.fillText('CURRENT FRAME', w - 6, NOW - 5);
    g.textAlign = 'left';
    g.fillStyle = C.faint;
    g.fillText('one token per entity per frame  ·  real recorded stream, segment ' + A.seg,
               6, h - 7);
  }

  /* ================= stage 1 — 3D pose episode =================
     A port of the demo film's act-2 panel (demo_video/pipe_act2_render.py): same geometry
     dump, same 42-degree camera orbiting the smoothed pair centre, same terrain wireframe
     fading in, same colours and depth fades. Verified against the film's own projection to
     sub-pixel agreement (see the port check in the repo). */
  var TERRAIN_IN = [2.35, 3.6];      // act-local seconds, as in the film

  function drawTerrain(g, P, cam, w, h, alpha) {
    var T = P.terrain, nz = T.z.length, nx = T.x.length, i, j;
    var pts = [], seen = [];
    for (i = 0; i < nz; i++) {
      pts.push([]);
      for (j = 0; j < nx; j++) {
        var hv = T.h[i][j];
        if (hv === -32768) { pts[i].push(null); continue; }      // unscanned cell
        var q = projF([T.x[j] / 100, hv / 100, T.z[i] / 100], cam, w, h);
        pts[i].push(q);
        if (q) seen.push(q[2]);
      }
    }
    if (!seen.length) return;
    seen.sort(function (p, q) { return p - q; });
    var dmin = seen[Math.floor(seen.length * 0.05)];
    g.lineWidth = 1;
    function seg(p, q) {
      if (!p || !q) return;
      var a = alpha * (1 - 0.75 * clamp(((p[2] + q[2]) / 2 - dmin) / 60, 0, 1));
      if (a <= 0.02) return;
      g.strokeStyle = fadeCol(RGB.harness, a * 0.55);
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
    }
    for (i = 0; i < nz; i++) for (j = 0; j < nx - 1; j++) seg(pts[i][j], pts[i][j + 1]);
    for (j = 0; j < nx; j++) for (i = 0; i < nz - 1; i++) seg(pts[i][j], pts[i + 1][j]);
  }

  function drawTree(g, pts, edges, rgb, dmin, dspan, dotR, wid) {
    var i;
    g.lineCap = 'round';
    for (i = 0; i < edges.length; i++) {
      var p = pts[edges[i][0]], q = pts[edges[i][1]];
      if (!p || !q) continue;
      g.strokeStyle = fadeCol(rgb, 1 - 0.55 * clamp(((p[2] + q[2]) / 2 - dmin) / dspan, 0, 1));
      g.lineWidth = wid;
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
    }
    for (i = 0; i < pts.length; i++) {
      if (!pts[i]) continue;
      g.fillStyle = fadeCol(rgb, 1 - 0.5 * clamp((pts[i][2] - dmin) / dspan, 0, 1));
      g.beginPath(); g.arc(pts[i][0], pts[i][1], dotR, 0, 6.2832); g.fill();
    }
  }

  function drawPose(g, w, h, t, settle) {
    var P = DATA.pose;
    if (!P || !P.cam) return;
    var dur = (P.monster.joints.length - 1) / P.fps;
    var tt = Math.min(t * settle, dur);                 // eases to a stop, never loops
    var cam = filmCam(P, tt);

    var aT = clamp((tt - TERRAIN_IN[0]) / (TERRAIN_IN[1] - TERRAIN_IN[0]), 0, 1);
    if (aT > 0.02) drawTerrain(g, P, cam, w, h, aT);

    var sets = [['monster', RGB.red, 1.5, 1.6], ['hunter', RGB.neural, 1.7, 1.9],
                ['weapon', RGB.gold, 1.4, 1.6]];
    var proj = {}, all = [];
    sets.forEach(function (sp) {
      if (!P[sp[0]]) return;
      proj[sp[0]] = jointsAt(P[sp[0]], tt, P.fps).map(function (p) { return projF(p, cam, w, h); });
      if (sp[0] !== 'weapon') proj[sp[0]].forEach(function (q) { if (q) all.push(q[2]); });
    });
    if (!all.length) return;
    var dmin = Math.min.apply(null, all);
    var dspan = Math.max(1e-3, Math.max.apply(null, all) - dmin + 8);
    sets.forEach(function (sp) {
      if (proj[sp[0]]) drawTree(g, proj[sp[0]], P[sp[0]].edges, sp[1], dmin, dspan, sp[2], sp[3]);
    });

    g.font = MONO_S; g.textAlign = 'left'; g.fillStyle = C.faint;
    g.fillText('276-D articulated world state, decoded to metric world joints  ·  ' +
               'monster 54  ·  hunter 32  ·  weapon 2', 6, h - 20);
    if (aT > 0.15) {
      g.fillStyle = fadeCol(RGB.harness, 0.9 * aT);
      g.fillText('+ static terrain: the surface the bodies are trained to respect', 6, h - 6);
    }
  }

  /* ================= stage 2 — the bridge, projecting live ================= */
  // Left: the state in 3D with the camera and its frustum. Right: what the fixed
  // operator produces from it, with the same channel encoding the real renderer
  // uses (height -> R, per-joint id -> G, inverse depth -> B).
  function drawBridge(g, w, h, t, settle) {
    var P = DATA.pose;
    if (!P) return;
    var n = P.monster.joints.length;
    var fi = Math.min(n - 1, Math.floor(t * P.fps * settle) % n);
    var split = w * 0.47;

    // ---- the render camera we are illustrating
    var az = 0.55 + 0.16 * Math.sin(t * 0.3);
    var rcam = makeCam(az, 0.22, 8.5, [0, 1.0, 0]);
    // ---- the observer camera, looking at the whole setup from outside
    var ocam = makeCam(-1.15, 0.42, 17, [0, 1.0, 0]);

    var rw = w - split - 14, rh = rw * 704 / 1280;
    var ry = (h - rh) / 2;

    // ================= left: 3D + frustum =================
    g.save();
    g.beginPath(); g.rect(0, 0, split - 6, h); g.clip();

    g.strokeStyle = 'rgba(236,231,219,0.06)';
    for (var i = -6; i <= 6; i += 2) {
      [[[i, 0, -6], [i, 0, 6]], [[-6, 0, i], [6, 0, i]]].forEach(function (s) {
        var a = project(s[0], ocam, split, h), b = project(s[1], ocam, split, h);
        if (a[2] > 0 && b[2] > 0) { g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); }
      });
    }

    // frustum from the render camera
    var reveal = easeInOut(clamp(t / 2.2, 0, 1));
    var corners = [[-1, -0.55], [1, -0.55], [1, 0.55], [-1, 0.55]].map(function (c) {
      var far = 6.5;
      var tanf = Math.tan(0.62 / 2);
      var p = [rcam.eye[0] + rcam.f[0] * far + rcam.r[0] * c[0] * far * tanf + rcam.u[0] * c[1] * far * tanf,
               rcam.eye[1] + rcam.f[1] * far + rcam.r[1] * c[0] * far * tanf + rcam.u[1] * c[1] * far * tanf,
               rcam.eye[2] + rcam.f[2] * far + rcam.r[2] * c[0] * far * tanf + rcam.u[2] * c[1] * far * tanf];
      return [rcam.eye[0] + (p[0] - rcam.eye[0]) * reveal,
              rcam.eye[1] + (p[1] - rcam.eye[1]) * reveal,
              rcam.eye[2] + (p[2] - rcam.eye[2]) * reveal];
    });
    var ce = project(rcam.eye, ocam, split, h);
    var cc = corners.map(function (p) { return project(p, ocam, split, h); });
    g.strokeStyle = C.gold;
    g.setLineDash([4, 4]);
    g.lineWidth = 1.1;
    g.globalAlpha = 0.75;
    cc.forEach(function (p) {
      if (ce[2] <= 0 || p[2] <= 0) return;
      g.beginPath(); g.moveTo(ce[0], ce[1]); g.lineTo(p[0], p[1]); g.stroke();
    });
    g.setLineDash([]);
    g.beginPath();
    cc.forEach(function (p, k) { if (p[2] > 0) { k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); } });
    g.closePath();
    g.stroke();
    g.globalAlpha = 1;
    g.fillStyle = C.gold;
    g.beginPath(); g.arc(ce[0], ce[1], 4, 0, 6.2832); g.fill();

    [['monster', C.gold], ['hunter', C.neural], ['weapon', C.harness]].forEach(function (sp) {
      var ent = P[sp[0]];
      if (!ent || !ent.joints[fi]) return;
      var pts = frameOf(ent, fi).map(function (p) { return project(p, ocam, split, h); });
      g.strokeStyle = sp[1]; g.lineWidth = 1.4; g.globalAlpha = 0.9; g.lineCap = 'round';
      ent.edges.forEach(function (e) {
        var a = pts[e[0]], b = pts[e[1]];
        if (!a || !b || a[2] <= 0 || b[2] <= 0) return;
        g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
      });
    });
    g.globalAlpha = 1;
    g.restore();
    g.fillStyle = C.faint; g.font = MONO_S; g.textAlign = 'left';
    g.fillText('STATE + CAMERA', 6, 16);

    // ================= right: the rasterized channels =================
    g.fillStyle = '#000';
    g.fillRect(split + 14, ry, rw, rh);

    // per-entity G bands, exactly as the real shader allocates them
    var G_BAND = { monster: [105, 194], hunter: [195, 254], weapon: [60, 99] };
    var order = ['monster', 'hunter', 'weapon'];
    var heights = [];
    order.forEach(function (k) {
      var ent = P[k];
      if (ent && ent.joints[fi]) frameOf(ent, fi).forEach(function (p) { heights.push(p[1]); });
    });
    var hlo = Math.min.apply(null, heights), hhi = Math.max.apply(null, heights) + 0.001;

    g.save();
    g.beginPath(); g.rect(split + 14, ry, rw, rh); g.clip();
    order.forEach(function (k) {
      var ent = P[k];
      if (!ent || !ent.joints[fi]) return;
      var W = frameOf(ent, fi);
      var pts = W.map(function (p) { return project(p, rcam, rw, rh); });
      var band = G_BAND[k];
      ent.edges.forEach(function (e, ei) {
        var a = pts[e[0]], b = pts[e[1]];
        if (!a || !b || a[2] <= 0 || b[2] <= 0) return;
        var hn = clamp(((W[e[0]][1] + W[e[1]][1]) / 2 - hlo) / (hhi - hlo), 0, 1);
        var z = (a[2] + b[2]) / 2;
        var idn = band[0] + (band[1] - band[0]) * (ei / Math.max(1, ent.edges.length - 1));
        // bone branch of the real encoding: R = 0.2*h, G = id/255, B = inverse depth
        var R = Math.round(255 * 0.2 * hn);
        var G = Math.round(idn);
        var B = Math.round(255 * clamp(1 - (z - 3) / 12, 0, 1));
        g.strokeStyle = 'rgb(' + R + ',' + G + ',' + B + ')';
        g.lineWidth = clamp(3.4 * (7 / z), 1.2, 7);
        g.lineCap = 'round';
        g.beginPath();
        g.moveTo(split + 14 + a[0], ry + a[1]);
        g.lineTo(split + 14 + b[0], ry + b[1]);
        g.stroke();
      });
    });
    g.restore();
    g.strokeStyle = 'rgba(236,231,219,0.18)';
    g.lineWidth = 1;
    g.strokeRect(split + 14.5, ry + 0.5, rw - 1, rh - 1);

    g.fillStyle = C.faint; g.font = MONO_S;
    g.fillText('POSE-CONTROL FRAME  ·  R height  G per-joint id  B inverse depth',
               split + 14, ry - 8);
    g.fillText('x = K [R | t] X   ·   closed form, zero learnable parameters', 6, h - 7);
  }

  /* ================= stage 3 — chunk relay ================= */
  function drawRelay(g, w, h, t, settle) {
    var NCH = 6, LEN = 81, SEED = 8;
    var pad = 18, gap = 10;
    var cw = (w - pad * 2 - gap * (NCH - 1)) / NCH;
    var ch = Math.min(72, h * 0.30);
    var y = h * 0.40;
    var per = 1.5;                                    // seconds per chunk
    var prog = t / per * settle;

    for (var i = 0; i < NCH; i++) {
      var x = pad + i * (cw + gap);
      var p = clamp(prog - i, 0, 1);
      g.strokeStyle = 'rgba(236,231,219,0.16)';
      g.lineWidth = 1;
      roundRect(g, x, y, cw, ch, 5);
      g.stroke();
      if (p > 0) {
        g.fillStyle = 'rgba(242,109,91,0.20)';
        roundRect(g, x, y, cw * p, ch, 5);
        g.fill();
      }
      // the seed frames handed over from the previous chunk
      if (i > 0 && p > 0) {
        g.fillStyle = C.harness;
        g.globalAlpha = 0.55;
        g.fillRect(x, y, cw * (SEED / LEN), ch);
        g.globalAlpha = 1;
      }
      g.fillStyle = p > 0 ? C.ink : C.faint;
      g.font = MONO_S;
      g.textAlign = 'center';
      g.fillText('chunk ' + i, x + cw / 2, y + ch + 15);
      if (p >= 1) {
        g.fillStyle = C.red;
        g.fillText(LEN + ' f', x + cw / 2, y - 8);
      }
    }
    // the relay arrows
    g.strokeStyle = C.harness;
    g.globalAlpha = 0.7;
    g.lineWidth = 1.2;
    for (var j = 0; j < NCH - 1; j++) {
      if (prog < j + 1) continue;
      var x0 = pad + j * (cw + gap) + cw, x1 = x0 + gap;
      var ym = y + ch + 26;
      g.beginPath();
      g.moveTo(x0 - cw * 0.12, y + ch);
      g.bezierCurveTo(x0, ym, x1, ym, x1 + cw * 0.02, y + ch);
      g.stroke();
    }
    g.globalAlpha = 1;
    g.textAlign = 'left';
    g.fillStyle = C.harness;
    g.font = MONO_S;
    g.fillText('last frames of each chunk seed the next  ·  appearance is what is relayed', pad, h - 22);
    g.fillStyle = C.faint;
    g.fillText('geometry is not relayed: every chunk is handed its own pose control', pad, h - 8);
  }

  var DRAW = [drawActions, drawPose, drawBridge, drawRelay];

  /* ---------------- driver ---------------- */
  var canvas, ctx, active = 0, t0 = 0, raf = null, dpr = 1;

  function resize() {
    if (!canvas) return;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    var r = canvas.getBoundingClientRect();
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }

  function tick(now) {
    if (!canvas) return;
    var t = (now - t0) / 1000;
    var w = canvas.width / dpr, h = canvas.height / dpr;
    // settle: 1 while running, easing to 0 over the last EASE_SECS, then hold
    var settle = t < RUN_SECS - EASE_SECS ? 1
      : Math.max(0, 1 - (t - (RUN_SECS - EASE_SECS)) / EASE_SECS);
    settle = settle * settle * (3 - 2 * settle);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.globalAlpha = 1;
    ctx.textBaseline = 'alphabetic';
    // A stage must never take the page down, but a swallowed error is worse: it once let
    // a stage that threw on every frame pass the test suite. Record and surface it.
    try {
      DRAW[active](ctx, w, h, t, settle);
    } catch (e) {
      window.StageAnim.errors.push({ stage: active, message: String(e && e.message || e) });
      if (window.console) console.error('stage ' + active + ' draw failed', e);
    }

    if (t < RUN_SECS + 0.2) raf = requestAnimationFrame(tick);
    else raf = null;
  }

  function play(i) {
    active = i;
    t0 = performance.now();
    if (raf) cancelAnimationFrame(raf);
    resize();
    raf = requestAnimationFrame(tick);
  }

  // exposed so the port can be diffed against the film's own projection
  window.__stageInternals = { filmCam: filmCam, jointsAt: jointsAt, projF: projF };
  window.StageAnim = {
    errors: [],
    init: function (el) {
      canvas = el;
      ctx = canvas.getContext('2d');
      resize();
      window.addEventListener('resize', function () { resize(); if (!raf) play(active); });
      Promise.all([
        fetch('static/data/actions.json').then(function (r) { return r.json(); }),
        fetch('static/data/pose.json').then(function (r) { return r.json(); })
      ]).then(function (v) {
        DATA.actions = v[0]; DATA.pose = v[1]; play(active);
      }).catch(function () {
        ctx.fillStyle = C.faint; ctx.font = MONO;
        ctx.fillText('could not load static/data/*.json', 12, 24);
      });
    },
    play: play
  };
})();
