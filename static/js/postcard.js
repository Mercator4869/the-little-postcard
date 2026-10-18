/* =====================================================================
   明信片翻面特效（匹配结果页）
   ----------------------------------------------------------------
   1. 基础层：翻面由 CSS 3D 完成（.postcard.is-flipped .postcard__inner）。
   2. 叠加层：本文件的 Canvas 盖在卡片之上，只做特效，不接管点击。
      · 手绘线条残影：canvas 从不清空，每帧用 destination-out 擦一层
        半透明，旧线条自己变淡 → 动画常见的运动残留/拖尾。
      · 闪光粒子：小十字 / 星光，从卡片四角与轮廓迸发，向外飘散。
   3. 线条要贴着「翻转中的」卡片，所以不自己算 3D 投影，而是读取
      四个角落标记元素的实际投影坐标（浏览器已经算好了），保证同步。
   ===================================================================== */
(function () {
  'use strict';

  var stage = document.querySelector('[data-postcard]');
  if (!stage) { return; }

  var card = stage.querySelector('.postcard');
  var canvas = stage.querySelector('[data-postcard-canvas]');
  var corners = stage.querySelectorAll('.postcard__corner');
  if (!card || !canvas || corners.length < 4) { return; }

  var ctx = canvas.getContext ? canvas.getContext('2d') : null;
  if (!ctx) { return; }

  var reduceMotion = !!(window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  var FLIP_MS = readFlipMs();
  var FADE_MS = 1150;        // 停止画线后再让残影淡出的时间
  var DECAY = 0.11;          // 每帧擦掉的透明度比例（越大残影越短）
  var MAX_PARTICLES = 260;
  var PALETTE = [
    'rgba(255, 240, 196, 1)',
    'rgba(255, 214, 120, 1)',
    'rgba(255, 255, 255, 1)',
    'rgba(206, 236, 255, 1)'
  ];

  var w = 0;
  var h = 0;
  var dpr = 1;
  var raf = 0;
  var last = 0;
  var emitUntil = 0;
  var clearAt = 0;
  var particles = [];
  var flipped = false;

  /* ------------------------------------------------------------ 工具 */

  function readFlipMs() {
    var raw = getComputedStyle(document.documentElement)
      .getPropertyValue('--pc-flip-ms').trim();
    var n = parseFloat(raw);
    if (isNaN(n)) { return 820; }
    return raw.indexOf('ms') === -1 ? n * 1000 : n;
  }

  function resize() {
    var rect = stage.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = Math.max(1, Math.round(rect.width));
    h = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
  }

  /* 四个角在 canvas 坐标系里的当前位置（含 3D 投影） */
  function readQuad() {
    var base = stage.getBoundingClientRect();
    var pts = [];
    for (var i = 0; i < 4; i++) {
      var r = corners[i].getBoundingClientRect();
      pts.push({
        x: r.left + r.width / 2 - base.left,
        y: r.top + r.height / 2 - base.top
      });
    }
    return pts; // tl, tr, br, bl
  }

  /* 把一条边采样成带抖动的折线，模拟手绘的毛边 */
  function scribbleEdge(a, b, amp, steps) {
    var pts = [];
    var dx = b.x - a.x;
    var dy = b.y - a.y;
    var len = Math.sqrt(dx * dx + dy * dy) || 1;
    var nx = -dy / len;
    var ny = dx / len;
    for (var i = 0; i <= steps; i++) {
      var t = i / steps;
      var bulge = Math.sin(t * Math.PI); // 两端贴合、中段抖动大
      var j = (Math.random() - 0.5) * 2 * amp * (0.35 + 0.65 * bulge);
      pts.push({
        x: a.x + dx * t + nx * j,
        y: a.y + dy * t + ny * j
      });
    }
    return pts;
  }

  function strokePath(pts, close) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (var i = 1; i < pts.length - 1; i++) {
      var mx = (pts[i].x + pts[i + 1].x) / 2;
      var my = (pts[i].y + pts[i + 1].y) / 2;
      ctx.quadraticCurveTo(pts[i].x, pts[i].y, mx, my);
    }
    var end = pts[pts.length - 1];
    ctx.lineTo(end.x, end.y);
    if (close) { ctx.closePath(); }
    ctx.stroke();
  }

  /* ------------------------------------------------------- 线条残影 */

  function drawTrail(quad, energy) {
    var amp = 2.2 + 3.4 * energy;
    var pts = [];
    for (var i = 0; i < 4; i++) {
      pts = pts.concat(scribbleEdge(quad[i], quad[(i + 1) % 4], amp, 8));
    }

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // ① 暖色光晕：给残影一点「发光」的体积
    ctx.globalAlpha = 0.12 + 0.11 * energy;
    ctx.lineWidth = 6 + 4 * energy;
    ctx.strokeStyle = 'rgba(255, 196, 92, 1)';
    ctx.shadowColor = 'rgba(255, 214, 130, 0.95)';
    ctx.shadowBlur = 12 + 8 * energy;
    strokePath(pts, true);

    // ② 亮白细线：动画里的速度线
    ctx.globalAlpha = 0.62 + 0.2 * energy;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 253, 244, 1)';
    ctx.shadowBlur = 8;
    strokePath(pts, true);

    // ③ 墨线核心：让线条在纸面上有手绘的分量
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 0.24 + 0.12 * energy;
    ctx.lineWidth = 1.1;
    ctx.strokeStyle = 'rgba(138, 90, 30, 1)';
    strokePath(pts, true);

    ctx.restore();
  }

  /* ----------------------------------------------------------- 粒子 */

  function spawn(x, y, vx, vy, size) {
    if (particles.length >= MAX_PARTICLES) { return; }
    var life = 42 + Math.random() * 46;
    particles.push({
      x: x,
      y: y,
      vx: vx,
      vy: vy,
      size: size,
      life: life,
      max: life,
      rot: Math.random() * Math.PI,
      spin: Math.random() * Math.PI * 2,
      alpha: 0.55 + Math.random() * 0.45,
      color: PALETTE[(Math.random() * PALETTE.length) | 0]
    });
  }

  function drawSparkle(p) {
    var left = p.life / p.max;                       // 1 → 0
    var twinkle = 0.4 + 0.6 * Math.abs(
      Math.sin(p.spin + (p.max - p.life) * 0.22)        // 一闪一闪
    );
    var a = Math.min(1, left * 1.35) * twinkle * p.alpha;
    if (a <= 0.01) { return; }
    var s = p.size * (0.55 + 0.75 * (1 - left));

    ctx.save();
    ctx.globalAlpha = a;
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.lineCap = 'round';
    ctx.strokeStyle = p.color;

    // 主十字
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(-s, 0); ctx.lineTo(s, 0);
    ctx.moveTo(0, -s); ctx.lineTo(0, s);
    ctx.stroke();

    // 斜向短线，凑成四角星光
    if (s > 3) {
      var d = s * 0.4;
      ctx.globalAlpha = a * 0.7;
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      ctx.moveTo(-d, -d); ctx.lineTo(d, d);
      ctx.moveTo(-d, d); ctx.lineTo(d, -d);
      ctx.stroke();
    }

    // 中心留一点高光
    ctx.globalAlpha = a * 0.9;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.beginPath();
    ctx.arc(0, 0, Math.max(0.7, s * 0.15), 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  function pointOnQuad(quad) {
    var edge = (Math.random() * 4) | 0;
    var a = quad[edge];
    var b = quad[(edge + 1) % 4];
    var t = Math.random();
    return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
  }

  /* 四个角同时迸发 */
  function burst(quad) {
    var cx = (quad[0].x + quad[1].x + quad[2].x + quad[3].x) / 4;
    var cy = (quad[0].y + quad[1].y + quad[2].y + quad[3].y) / 4;
    for (var c = 0; c < 4; c++) {
      var x = quad[c].x;
      var y = quad[c].y;
      var dx = x - cx;
      var dy = y - cy;
      var d = Math.sqrt(dx * dx + dy * dy) || 1;
      var base = Math.atan2(dy / d, dx / d);
      for (var k = 0; k < 9; k++) {
        var ang = base + (Math.random() - 0.5) * 1.6;
        var sp = 0.7 + Math.random() * 2.6;
        spawn(x, y, Math.cos(ang) * sp, Math.sin(ang) * sp,
              3 + Math.random() * 4);
      }
    }
  }

  /* ------------------------------------------------------------ 主循环 */

  function frame(now) {
    if (!last) { last = now; }
    var dt = Math.min((now - last) / 16.6667, 2.4); // 归一化成「帧」
    last = now;

    // 1) 擦一层半透明底：画布从不整体清空，旧线自己变淡 = 拖尾
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0, 0, 0, ' + (DECAY * dt) + ')';
    ctx.fillRect(0, 0, w, h);
    ctx.restore();

    // 2) 翻转过程中沿卡片轮廓画线 + 撒星光
    if (now < emitUntil) {
      var quad = readQuad();
      var progress = 1 - (emitUntil - now) / FLIP_MS;
      var energy = Math.max(0, 1 - Math.abs(progress - 0.5) * 1.7);

      drawTrail(quad, energy);

      var n = Math.random() < 0.6 ? 1 : 2;
      for (var i = 0; i < n; i++) {
        var p = pointOnQuad(quad);
        var ang = Math.random() * Math.PI * 2;
        var sp = 0.4 + Math.random() * 1.8;
        spawn(p.x, p.y, Math.cos(ang) * sp, Math.sin(ang) * sp - 0.45,
              2 + Math.random() * 3.4);
      }
    }

    // 3) 粒子
    for (var j = particles.length - 1; j >= 0; j--) {
      var q = particles[j];
      q.x += q.vx * dt;
      q.y += q.vy * dt;
      q.vx *= 1 - 0.022 * dt;
      q.vy = q.vy * (1 - 0.022 * dt) - 0.012 * dt; // 轻微上飘
      q.life -= dt;
      if (q.life <= 0 || q.x < -70 || q.x > w + 70 ||
          q.y < -70 || q.y > h + 70) {
        particles.splice(j, 1);
        continue;
      }
      drawSparkle(q);
    }

    // 翻转结束、残影淡完 → 收工并清干净
    if (now > clearAt && particles.length === 0) {
      ctx.clearRect(0, 0, w, h);
      raf = 0;
      last = 0;
      return;
    }

    raf = window.requestAnimationFrame(frame);
  }

  function startEffects() {
    if (reduceMotion) { return; }
    resize();
    emitUntil = performance.now() + FLIP_MS;
    clearAt = emitUntil + FADE_MS;
    burst(readQuad());
    if (!raf) {
      last = 0;
      raf = window.requestAnimationFrame(frame);
    }
  }

  /* ------------------------------------------------------------- 交互 */

  function toggle() {
    setFlipped(!flipped);
  }

  function setFlipped(value) {
    if (value === flipped) { return; }
    flipped = value;
    card.classList.toggle('is-flipped', flipped);
    card.setAttribute('aria-pressed', flipped ? 'true' : 'false');
    startEffects();
    // 翻到笔记那面就把这首歌唱起来；之后一直放，翻回正面也不打断，
    // 想停用下面的原生播放器即可。
    if (flipped) {
      revealDetails();
      playSong();
    }
  }

  /* 上传页用的是「按钮翻面」：卡片里全是输入框，点卡片翻会打架 */
  var manualFlip = stage.getAttribute('data-postcard-flip') === 'manual';

  if (manualFlip) {
    var toggles = document.querySelectorAll('[data-postcard-toggle]');
    for (var t = 0; t < toggles.length; t++) {
      toggles[t].addEventListener('click', function (ev) {
        ev.preventDefault();
        toggle();
      });
    }
    card.classList.add('postcard--manual');
  } else {
    card.addEventListener('click', function () {
      // 反面笔记允许选中复制，选中文字时不翻面
      var sel = window.getSelection ? String(window.getSelection()) : '';
      if (sel.length > 0) { return; }
      toggle();
    });

    card.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' || ev.key === ' ' || ev.key === 'Spacebar') {
        ev.preventDefault();
        toggle();
      }
    });
  }

  /* 给同页的其他脚本（如 compose.js）用 */
  window.DSHPostcard = {
    toggle: toggle,
    setFlipped: setFlipped,
    isFlipped: function () { return flipped; }
  };

  /* --------------------------------------------------------- 尺寸维护 */

  var resizeTimer = 0;
  function scheduleResize() {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(function () {
      resize();
      fitNoteText();
    }, 120);
  }

  if (window.ResizeObserver) {
    new window.ResizeObserver(scheduleResize).observe(stage);
  } else {
    window.addEventListener('resize', scheduleResize);
  }

  /* ------------------------------------------------- 反面笔记的排版 */

  var noteEl = stage.querySelector('.pc-back__note');
  var noteBody = stage.querySelector('.pc-back__note-body');
  /* 从大往小收：短笔记就写成大字，长笔记收到下限后改为滚动 */
  var NOTE_MAX = 26;
  var NOTE_MIN = 13.5;

  /* 笔记字数不定，先按最大字号排，排不下再往下收 */
  function fitNoteText() {
    if (!noteEl) { return; }
    noteEl.classList.remove('is-short');
    var size = NOTE_MAX;
    noteEl.style.fontSize = size + 'px';
    var guard = 0;
    while (noteEl.scrollHeight > noteEl.clientHeight + 1 &&
           size > NOTE_MIN && guard < 40) {
      size -= 0.5;
      noteEl.style.fontSize = size + 'px';
      guard += 1;
    }
    // scrollHeight 不会小于 clientHeight，所以内容实际高度要看内层元素；
    // 明显没占满时垂直居中，避免整张卡片头重脚轻。
    var contentHeight = noteBody
      ? noteBody.getBoundingClientRect().height
      : noteEl.scrollHeight;
    if (contentHeight < noteEl.clientHeight - 24) {
      noteEl.classList.add('is-short');
    }
    markNoteOverflow();
  }

  /* 反面笔记太长时，底部渐隐提示「还能往下读」 */
  function markNoteOverflow() {
    if (!noteEl) { return; }
    noteEl.classList.toggle(
      'is-overflowing',
      noteEl.scrollHeight - noteEl.clientHeight > 2
    );
    noteEl.classList.toggle(
      'is-at-bottom',
      noteEl.scrollTop + noteEl.clientHeight >= noteEl.scrollHeight - 4
    );
  }

  if (noteEl) {
    noteEl.addEventListener('scroll', markNoteOverflow, { passive: true });
  }

  /* ------------------------------------------------------- 翻面就放歌 */

  var audio = document.getElementById('player');
  var audioHint = document.getElementById('audio-hint');
  var details = document.querySelector('[data-result-details]');
  var hintText = document.querySelector('.postcard-hint');

  function playSong() {
    if (!audio) { return; }
    var attempt = audio.play();
    if (attempt && typeof attempt.then === 'function') {
      attempt.then(function () {
        if (audioHint) { audioHint.hidden = true; }
      }, function () {
        // 手机浏览器常拦自动播放：提示用户点显式按钮，不静默失败
        if (audioHint) {
          audioHint.textContent = '浏览器阻止了自动播放，请点上面的「播放这首歌」按钮。';
          audioHint.hidden = false;
        }
      });
    }
  }

  var playButtons = document.querySelectorAll('[data-play-song]');
  for (var b = 0; b < playButtons.length; b++) {
    playButtons[b].addEventListener('click', function (event) {
      event.preventDefault();
      playSong();
    });
  }

  // 命中的明信片图加载失败时退回用户自己上传的图，避免手机上一片空白
  var frontImage = document.querySelector('.pc-front img[data-fallback-src]');
  if (frontImage) {
    frontImage.addEventListener('error', function () {
      var fallback = frontImage.getAttribute('data-fallback-src');
      if (fallback && frontImage.getAttribute('src') !== fallback) {
        frontImage.setAttribute('src', fallback);
      }
    });
  }

  /* 第一次翻面才把下面的歌名、歌手、播放器放出来 */
  function revealDetails() {
    if (!details || !details.hidden) { return; }
    details.hidden = false;
    if (hintText) {
      hintText.textContent = '再轻触明信片，可以翻回正面';
    }
  }

  resize();
  fitNoteText();
}());
