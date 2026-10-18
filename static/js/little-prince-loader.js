/* ============================================================
   The Little Postcard — reusable loading overlay controller
   Namespaced for the Image2Song site. No third-party dependencies.
   ============================================================ */

(function () {
  "use strict";

  var root = document.getElementById("lpLoader");
  if (!root) { return; }

  var canvas = document.getElementById("lp-starfall");
  var stillStarsSvg = document.getElementById("lp-stillStars");
  var progressEl = document.getElementById("lp-progress");
  var fillPath = document.getElementById("lp-progressFill");
  var starMarker = document.getElementById("lp-progressStar");
  var percentEl = document.getElementById("lp-percent");
  var statusEl = document.getElementById("lp-status");
  var hudCopy = document.getElementById("lp-hudCopy");

  var reducedMotion = !!(window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  var COPY_DEFAULT = "正在把这张明信片，寄往属于它的歌里";
  var COPY_DONE = "明信片已抵达，属于它的歌正在展开";
  var STATUS_STEPS = [
    [0, "正在折好信纸"],
    [18, "正在画上一双翅膀"],
    [40, "正在等一阵风"],
    [63, "正在穿过星星"],
    [84, "正在寻找那首歌"],
    [96, "就快到了"]
  ];

  var ctx = canvas ? canvas.getContext("2d") : null;
  var isOpen = false;
  var starsRaf = 0;
  var starsLast = 0;
  var fillLength = 0;
  var measured = false;
  var hideTimer = 0;
  var bodyLocked = false;

  var stars = [];
  var fixedStars = [];
  var meteors = [];
  var nextMeteor = 0;
  var width = 0;
  var height = 0;
  var dpr = 1;

  var PALETTE = [
    [255, 214, 112],
    [255, 231, 154],
    [230, 188, 90],
    [255, 242, 191],
    [210, 170, 79]
  ];

  var FIXED_PALETTE = [
    [250, 239, 206],
    [247, 226, 169],
    [229, 240, 245],
    [237, 211, 136]
  ];

  /* ------------------------------------------------------------ progress */

  function measureLine() {
    if (!fillPath) { return; }
    fillLength = fillPath.getTotalLength();
    fillPath.style.strokeDasharray = String(fillLength);
    fillPath.style.strokeDashoffset = String(fillLength);
    measured = true;
  }

  function statusFor(value) {
    var label = STATUS_STEPS[0][1];
    for (var i = 0; i < STATUS_STEPS.length; i++) {
      if (value >= STATUS_STEPS[i][0]) { label = STATUS_STEPS[i][1]; }
    }
    return label;
  }

  function setStatus(text) {
    if (!statusEl || !text || statusEl.textContent === text) { return; }
    statusEl.textContent = text;
    if (!reducedMotion && statusEl.animate) {
      statusEl.animate(
        [
          { opacity: 0, transform: "translateY(2px)" },
          { opacity: 1, transform: "translateY(0)" }
        ],
        { duration: 380, easing: "ease" }
      );
    }
  }

  function setProgress(value, statusText) {
    var p = Math.max(0, Math.min(100, Number(value) || 0));
    var label = statusText || statusFor(p);

    if (!measured) { measureLine(); }

    root.style.setProperty("--lp-progress", String(p / 100));

    if (progressEl) {
      progressEl.setAttribute("aria-valuenow", String(Math.round(p)));
      progressEl.setAttribute("aria-valuetext", Math.round(p) + "% · " + label);
    }
    if (percentEl) {
      percentEl.textContent = String(Math.floor(p)).padStart(2, "0");
    }
    if (fillPath && fillLength) {
      fillPath.style.strokeDashoffset = String(fillLength * (1 - p / 100));
    }
    if (starMarker && fillPath && fillLength) {
      var point = fillPath.getPointAtLength(fillLength * (p / 100));
      starMarker.setAttribute("transform", "translate(" + point.x.toFixed(2) + " " + point.y.toFixed(2) + ")");
      starMarker.style.opacity = p > 1 ? "1" : "0";
    }

    setStatus(label);
    return p;
  }

  /* ------------------------------------------------------------ host page */

  function setHostInert(on) {
    var nodes = [
      document.querySelector(".site-header"),
      document.querySelector("body > main"),
      document.querySelector(".site-footer")
    ];
    for (var i = 0; i < nodes.length; i++) {
      if (!nodes[i]) { continue; }
      if (on) { nodes[i].setAttribute("inert", ""); }
      else { nodes[i].removeAttribute("inert"); }
    }
  }

  function lockHost() {
    if (bodyLocked) { return; }
    document.body.classList.add("lp-loader-open");
    setHostInert(true);
    bodyLocked = true;
  }

  function unlockHost() {
    if (!bodyLocked) { return; }
    document.body.classList.remove("lp-loader-open");
    setHostInert(false);
    bodyLocked = false;
  }

  /* ------------------------------------------------------------ stars */

  function makeStars() {
    var count = Math.max(70, Math.min(175, Math.round((width * height) / 8500)));
    stars = [];
    fixedStars = [];
    meteors = [];

    for (var i = 0; i < count; i++) {
      var depth = Math.pow(Math.random(), 1.35);
      var color = PALETTE[Math.floor(Math.random() * PALETTE.length)];
      stars.push({
        x: width * (0.12 + Math.random() * 0.76),
        y: height * (0.06 + Math.random() * 0.88),
        depth: depth,
        r: 0.45 + (1.5 - 0.45) * depth,
        speed: (42 + (330 - 42) * depth) * (0.75 + Math.random() * 0.6),
        trail: (7 + (86 - 7) * depth) * (0.7 + Math.random() * 0.55),
        alpha: (0.3 + (0.9 - 0.3) * depth) * (0.72 + Math.random() * 0.28),
        color: color,
        phase: Math.random() * Math.PI * 2,
        twinkle: 0.5 + Math.random() * 1.3,
        bend: (Math.random() - 0.5) * 6,
        sparkle: Math.random() > 0.8
      });
    }

    for (var j = 0; j < 30; j++) {
      fixedStars.push({
        x: width * (0.15 + Math.random() * 0.7),
        y: height * (0.1 + Math.random() * 0.74),
        r: 0.4 + Math.random() * 1.4,
        alpha: 0.14 + Math.random() * 0.34,
        phase: Math.random() * Math.PI * 2,
        twinkle: 0.3 + Math.random() * 1.2,
        color: FIXED_PALETTE[Math.floor(Math.random() * FIXED_PALETTE.length)],
        sparkle: Math.random() > 0.55
      });
    }
  }

  function resizeStars() {
    if (!canvas || !ctx) { return; }
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    makeStars();
  }

  function drawPencilStar(x, y, radius, alpha, color, angle) {
    var r = color[0], g = color[1], b = color[2];
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle || 0);
    ctx.strokeStyle = "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
    ctx.lineWidth = 0.65;
    ctx.lineCap = "round";

    var jitter = radius * 0.24;
    ctx.beginPath();
    ctx.moveTo(0, -radius);
    ctx.quadraticCurveTo(jitter, -radius * 0.25, 0, radius);
    ctx.moveTo(-radius, jitter * 0.4);
    ctx.quadraticCurveTo(-radius * 0.2, 0, radius, -jitter * 0.35);
    ctx.stroke();

    if (radius > 1.2) {
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.moveTo(-radius * 0.52, -radius * 0.52);
      ctx.lineTo(radius * 0.52, radius * 0.52);
      ctx.moveTo(radius * 0.52, -radius * 0.52);
      ctx.lineTo(-radius * 0.52, radius * 0.52);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawMeteors(now, delta) {
    if (now > nextMeteor) {
      nextMeteor = now + 1400 + Math.random() * 2400;
      var leftSide = Math.random() > 0.5;
      meteors.push({
        x: width * (leftSide ? 0.17 + Math.random() * 0.25 : 0.6 + Math.random() * 0.22),
        y: -80 - Math.random() * 80,
        vy: 440 + Math.random() * 280,
        vx: (Math.random() - 0.5) * 55,
        life: 0,
        maxLife: 1.5 + Math.random() * 0.9,
        trail: 110 + Math.random() * 100,
        gold: Math.random() > 0.58
      });
    }

    meteors = meteors.filter(function (m) {
      m.life += delta;
      m.x += m.vx * delta;
      m.y += m.vy * delta;
      if (m.life > m.maxLife || m.y - m.trail > height + 30) { return false; }

      var fade = Math.sin(Math.max(0, Math.min(1, m.life / m.maxLife)) * Math.PI);
      var color = m.gold ? [249, 222, 147] : [225, 192, 108];
      var cr = color[0], cg = color[1], cb = color[2];
      var headX = m.x, headY = m.y;
      var tailX = m.x - m.vx * 0.26;
      var tailY = m.y - m.trail;

      var grad = ctx.createLinearGradient(headX, headY, tailX, tailY);
      grad.addColorStop(0, "rgba(" + cr + ", " + cg + ", " + cb + ", " + (0.68 * fade) + ")");
      grad.addColorStop(0.32, "rgba(" + cr + ", " + cg + ", " + cb + ", " + (0.3 * fade) + ")");
      grad.addColorStop(1, "rgba(" + cr + ", " + cg + ", " + cb + ", 0)");
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.45;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(headX, headY);
      ctx.quadraticCurveTo(headX + m.vx * 0.5, headY - m.trail * 0.48, tailX, tailY);
      ctx.stroke();

      ctx.fillStyle = "rgba(255, 224, 145, " + (0.09 * fade) + ")";
      ctx.beginPath();
      ctx.arc(headX, headY, 12, 0, Math.PI * 2);
      ctx.fill();
      drawPencilStar(headX, headY, m.gold ? 4.2 : 3.6, 0.9 * fade, color, 0.2);
      return true;
    });
  }

  function drawStars(now, delta) {
    if (!ctx) { return; }
    ctx.clearRect(0, 0, width, height);

    var seconds = now / 1000;

    for (var i = 0; i < fixedStars.length; i++) {
      var fixed = fixedStars[i];
      var flicker = 0.55 + 0.45 * Math.sin(seconds * fixed.twinkle + fixed.phase);
      var alpha = fixed.alpha * flicker;
      if (alpha <= 0.015) { continue; }
      if (fixed.sparkle) {
        drawPencilStar(fixed.x, fixed.y, fixed.r * 2.7, alpha, fixed.color, fixed.phase * 0.1);
      } else {
        ctx.fillStyle = "rgba(" + fixed.color[0] + ", " + fixed.color[1] + ", " + fixed.color[2] + ", " + alpha + ")";
        ctx.beginPath();
        ctx.arc(fixed.x, fixed.y, Math.max(0.4, fixed.r * 0.72), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    for (var j = 0; j < stars.length; j++) {
      var star = stars[j];
      if (!reducedMotion) {
        star.y += star.speed * delta;
        if (star.y - star.trail > height + 12) {
          star.y = -12 - Math.random() * 40;
          star.x = width * (0.12 + Math.random() * 0.76);
        }
      }

      var fl = 0.72 + 0.28 * Math.sin(seconds * star.twinkle + star.phase);
      var al = Math.max(0, Math.min(0.8, star.alpha * fl));
      var s = star;
      var tailX = s.x + s.bend;
      var tailY = s.y - s.trail;

      var grad2 = ctx.createLinearGradient(s.x, s.y, tailX, tailY);
      grad2.addColorStop(0, "rgba(" + s.color[0] + ", " + s.color[1] + ", " + s.color[2] + ", " + (al * 0.7) + ")");
      grad2.addColorStop(0.5, "rgba(" + s.color[0] + ", " + s.color[1] + ", " + s.color[2] + ", " + (al * 0.26) + ")");
      grad2.addColorStop(1, "rgba(" + s.color[0] + ", " + s.color[1] + ", " + s.color[2] + ", 0)");
      ctx.strokeStyle = grad2;
      ctx.lineWidth = Math.max(0.45, s.r * 0.58);
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.quadraticCurveTo(s.x + s.bend * 0.6, s.y - s.trail * 0.52, tailX, tailY);
      ctx.stroke();

      if (s.depth > 0.58) {
        ctx.fillStyle = "rgba(255, 226, 151, " + (al * 0.07) + ")";
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r * 6.4, 0, Math.PI * 2);
        ctx.fill();
      }

      if (s.sparkle || s.r > 0.78) {
        drawPencilStar(s.x, s.y, s.r * 3.5, al, s.color, s.phase * 0.12);
      } else {
        ctx.fillStyle = "rgba(" + s.color[0] + ", " + s.color[1] + ", " + s.color[2] + ", " + al + ")";
        ctx.beginPath();
        ctx.arc(s.x, s.y, Math.max(0.35, s.r * 0.8), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (!reducedMotion) { drawMeteors(now, delta); }
  }

  function startStars() {
    if (!canvas || !ctx || starsRaf) { return; }
    resizeStars();
    starsLast = performance.now();
    nextMeteor = 0;

    if (reducedMotion) {
      drawStars(performance.now(), 0);
      return;
    }

    function frame(now) {
      var delta = Math.min((now - starsLast) / 1000, 0.05);
      starsLast = now;
      drawStars(now, delta);
      starsRaf = requestAnimationFrame(frame);
    }
    starsRaf = requestAnimationFrame(frame);
  }

  function stopStars() {
    if (starsRaf) {
      cancelAnimationFrame(starsRaf);
      starsRaf = 0;
    }
    if (ctx) {
      ctx.clearRect(0, 0, width, height);
    }
  }

  /* ------------------------------------------------------------ still paper stars */

  function makeStillStars() {
    if (!stillStarsSvg) { return; }
    var seed = 0x9e3779b9;
    var random = function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    var colors = ["#f6e7bd", "#ffe7ab", "#dbe8f3", "#edcf91", "#f9f1dd"];
    var parts = [];
    var i, x, y, color, alpha, dot, r;

    for (i = 0; i < 135; i++) {
      x = random() * 1440;
      y = random() * 900;
      if (y < 78) { y += 78; }
      if (y > 735 && x > 390 && x < 1050) { y = 680 + random() * 38; }
      color = colors[Math.floor(random() * colors.length)];
      alpha = 0.22 + random() * 0.55;
      dot = random() > 0.7;
      if (dot) {
        r = 0.55 + random() * 1.5;
        parts.push('<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + r.toFixed(2) + '" fill="' + color + '" opacity="' + alpha.toFixed(2) + '"/>');
      } else {
        r = 1.2 + random() * 3.2;
        parts.push(
          '<path d="M ' + x.toFixed(1) + " " + (y - r).toFixed(1) +
          " L " + (x + r * 0.25).toFixed(1) + " " + (y - r * 0.25).toFixed(1) +
          " L " + (x + r).toFixed(1) + " " + y.toFixed(1) +
          " L " + (x + r * 0.25).toFixed(1) + " " + (y + r * 0.25).toFixed(1) +
          " L " + x.toFixed(1) + " " + (y + r).toFixed(1) +
          " L " + (x - r * 0.25).toFixed(1) + " " + (y + r * 0.25).toFixed(1) +
          " L " + (x - r).toFixed(1) + " " + y.toFixed(1) +
          " L " + (x - r * 0.25).toFixed(1) + " " + (y - r * 0.25).toFixed(1) +
          ' Z" fill="' + color + '" opacity="' + alpha.toFixed(2) + '"/>'
        );
      }
    }

    var anchors = [
      [122, 158, 7], [1332, 378, 5], [1118, 722, 6], [322, 742, 4.6],
      [148, 604, 3.8], [1264, 136, 4.2], [998, 790, 3.4], [236, 350, 3.2],
      [1380, 664, 4], [70, 86, 3.3]
    ];
    for (i = 0; i < anchors.length; i++) {
      x = anchors[i][0]; y = anchors[i][1]; r = anchors[i][2];
      alpha = 0.55 + random() * 0.3;
      parts.push(
        '<g opacity="' + alpha.toFixed(2) + '" fill="none" stroke="#f4e1b0" stroke-width="0.8" stroke-linecap="round">' +
          '<path d="M ' + x + " " + (y - r) + " L " + (x + r * 0.22) + " " + (y - r * 0.22) + " L " + (x + r) + " " + y + " L " + (x + r * 0.22) + " " + (y + r * 0.22) + " L " + x + " " + (y + r) + " L " + (x - r * 0.22) + " " + (y + r * 0.22) + " L " + (x - r) + " " + y + " L " + (x - r * 0.22) + " " + (y - r * 0.22) + ' Z"/>' +
          '<path d="M ' + (x - r * 1.9) + " " + y + " L " + (x - r * 1.25) + " " + y + '" opacity="0.5"/>' +
          '<path d="M ' + (x + r * 1.25) + " " + y + " L " + (x + r * 1.9) + " " + y + '" opacity="0.5"/>' +
          '<path d="M ' + x + " " + (y - r * 1.9) + " L " + x + " " + (y - r * 1.25) + '" opacity="0.5"/>' +
          '<path d="M ' + x + " " + (y + r * 1.25) + " L " + x + " " + (y + r * 1.9) + '" opacity="0.5"/>' +
        "</g>"
      );
    }

    stillStarsSvg.innerHTML = parts.join("");
  }

  /* ------------------------------------------------------------ lifecycle */

  function resetCopy() {
    if (hudCopy) { hudCopy.textContent = COPY_DEFAULT; }
    root.classList.remove("lp-is-complete");
  }

  function showAt(value, statusText) {
    clearTimeout(hideTimer);
    measureLine();

    document.documentElement.classList.remove("lp-loader-pending");
    root.classList.add("lp-is-visible");
    root.classList.remove("lp-is-leaving", "lp-is-complete");
    root.setAttribute("aria-hidden", "false");
    root.setAttribute("aria-busy", "true");
    lockHost();
    resetCopy();

    if (!isOpen) {
      isOpen = true;
      startStars();
    } else {
      startStars();
    }

    setProgress(value == null ? 8 : value, statusText);
  }

  function finish() {
    root.classList.add("lp-is-complete");
    root.setAttribute("aria-busy", "false");
    if (hudCopy) { hudCopy.textContent = COPY_DONE; }
    setProgress(100, "信已抵达星星");
  }

  function hide() {
    if (!root.classList.contains("lp-is-visible")) { return; }
    root.classList.remove("lp-is-visible");
    root.classList.add("lp-is-leaving");
    root.setAttribute("aria-busy", "false");
    unlockHost();

    hideTimer = setTimeout(function () {
      root.classList.remove("lp-is-leaving", "lp-is-complete");
      root.setAttribute("aria-hidden", "true");
      stopStars();
      isOpen = false;
    }, reducedMotion ? 0 : 900);
  }

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && isOpen) { hide(); }
  });

  window.DSHLittlePrinceLoader = {
    showAt: showAt,
    setProgress: setProgress,
    finish: finish,
    hide: hide,
    setCopy: function (text) {
      if (hudCopy) { hudCopy.textContent = text; }
    },
    get isOpen() { return isOpen; }
  };

  makeStillStars();
  measureLine();
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () {
      if (!measured) { measureLine(); }
    });
  }

  var resizeTimer = 0;
  window.addEventListener("resize", function () {
    if (!isOpen) { return; }
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      resizeStars();
    }, 160);
  });

  var hiddenAt = 0;
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      hiddenAt = performance.now();
    } else if (hiddenAt && isOpen) {
      starsLast = performance.now();
      hiddenAt = 0;
    }
  });
})();
