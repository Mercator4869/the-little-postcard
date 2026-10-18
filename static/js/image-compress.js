/* ============================================================
   Compress large photos before upload.

   Why: phone photos are often 3–10 MB. Some free tunnels / slow
   links cap request bodies around a few hundred KB, which makes
   the upload POST fail while the loading overlay is still showing.
   Resizing to <=1600px and re-encoding as JPEG keeps the photo
   usable for CLIP matching and makes the upload reliable.
   ============================================================ */
(function () {
  "use strict";

  var form = document.querySelector("[data-postcard-form]");
  if (!form) { return; }

  var input = form.querySelector('input[type="file"][name="image"]');
  if (!input) { return; }

  var MAX_BYTES = 120 * 1024;   // only compress when larger than this
  var TARGET_BYTES = 95 * 1024; // free-tunnel request bodies get flaky above ~150 KB
  var MAX_EDGE = 1400;

  function loadImage(file) {
    return new Promise(function (resolve, reject) {
      var url = URL.createObjectURL(file);
      var img = new Image();
      img.onload = function () {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = function () {
        URL.revokeObjectURL(url);
        reject(new Error("image decode failed"));
      };
      img.src = url;
    });
  }

  function canvasToBlob(canvas, quality) {
    return new Promise(function (resolve, reject) {
      canvas.toBlob(function (blob) {
        if (blob) { resolve(blob); } else { reject(new Error("toBlob failed")); }
      }, "image/jpeg", quality);
    });
  }

  function drawScaled(img, edge) {
    var ratio = Math.min(1, edge / Math.max(img.naturalWidth, img.naturalHeight));
    var width = Math.max(1, Math.round(img.naturalWidth * ratio));
    var height = Math.max(1, Math.round(img.naturalHeight * ratio));
    var canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    return canvas;
  }

  function rename(file) {
    var base = (file.name || "photo").replace(/\.[^.]+$/, "");
    return base + ".jpg";
  }

  function compress(file) {
    return loadImage(file).then(function (img) {
      var edges = [MAX_EDGE, 1100, 850, 650];
      var qualities = [0.78, 0.65, 0.52, 0.42];

      function attempt(ei, qi) {
        if (ei >= edges.length) {
          return canvasToBlob(drawScaled(img, 560), 0.35).then(function (blob) {
            return new File([blob], rename(file), { type: "image/jpeg" });
          });
        }
        var canvas = drawScaled(img, edges[ei]);
        return canvasToBlob(canvas, qualities[qi]).then(function (blob) {
          if (blob.size <= TARGET_BYTES) {
            return new File([blob], rename(file), { type: "image/jpeg" });
          }
          if (qi < qualities.length - 1) {
            return attempt(ei, qi + 1);
          }
          return attempt(ei + 1, 0);
        });
      }

      return attempt(0, 0);
    });
  }

  form.addEventListener("submit", function (event) {
    var file = input.files && input.files[0];
    if (!file || file.size <= MAX_BYTES) { return; }
    if (input.dataset.compressing === "1") { return; }

    event.preventDefault();
    event.stopImmediatePropagation();
    input.dataset.compressing = "1";

    compress(file).then(function (out) {
      try {
        var transfer = new DataTransfer();
        transfer.items.add(out);
        input.files = transfer.files;
      } catch (error) {
        // 浏览器不支持 DataTransfer 时退回原图
      }
      input.dataset.compressing = "";
      if (typeof form.requestSubmit === "function") {
        form.requestSubmit();
      } else {
        HTMLFormElement.prototype.submit.call(form);
      }
    }).catch(function () {
      input.dataset.compressing = "";
      if (typeof form.requestSubmit === "function") {
        form.requestSubmit();
      } else {
        HTMLFormElement.prototype.submit.call(form);
      }
    });
  }, true);
})();
