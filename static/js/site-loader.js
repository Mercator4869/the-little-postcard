/* ============================================================
   Little Prince loading transition for the Image2Song site
   - upload → POST: overlay stays until the server answers
   - result: overlay continues, waits for the matched image, then fades
   ============================================================ */

(function () {
  "use strict";

  var PENDING_KEY = "lpLoaderPending";
  var PENDING_CLASS = "lp-loader-pending";
  var loader = window.DSHLittlePrinceLoader;

  if (!loader) { return; }

  window.addEventListener("pageshow", function (event) {
    if (event.persisted && loader.isOpen) { loader.hide(); }
  });

  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  function setPending() {
    try { sessionStorage.setItem(PENDING_KEY, "1"); } catch (error) {}
    document.documentElement.classList.add(PENDING_CLASS);
  }

  function clearPending() {
    try { sessionStorage.removeItem(PENDING_KEY); } catch (error) {}
    document.documentElement.classList.remove(PENDING_CLASS);
  }

  function hasPending() {
    try { return sessionStorage.getItem(PENDING_KEY) === "1"; } catch (error) {
      return document.documentElement.classList.contains(PENDING_CLASS);
    }
  }

  function startUploadOverlay() {
    var current = 8;
    loader.showAt(current, "正在等一阵风");

    var timer = window.setInterval(function () {
      if (!loader.isOpen) {
        window.clearInterval(timer);
        return;
      }
      current += Math.max(0.12, (88 - current) * 0.035);
      if (current > 87.5) { current = 87.5; }
      loader.setProgress(current);
    }, 220);

    // Keep a handle on the window in case the page is restored from bfcache.
    window.addEventListener("pagehide", function () { window.clearInterval(timer); }, { once: true });
    window.addEventListener("pageshow", function (event) {
      if (event.persisted) { loader.hide(); }
    }, { once: true });
  }

  function waitForImage(timeoutMs) {
    return new Promise(function (resolve) {
      var image = document.querySelector(".result-card .postcard__face--front img");
      if (!image) { resolve(); return; }

      var done = false;
      var timer = window.setTimeout(function () { finish(); }, timeoutMs);

      function finish() {
        if (done) { return; }
        done = true;
        window.clearTimeout(timer);
        resolve();
      }

      if (image.complete && image.naturalWidth) {
        if (image.decode) {
          image.decode().catch(function () {}).then(finish);
        } else {
          finish();
        }
      } else {
        image.addEventListener("load", finish, { once: true });
        image.addEventListener("error", finish, { once: true });
      }
    });
  }

  function startResultOverlay() {
    loader.showAt(84, "正在穿过最后一片星星");
    clearPending();

    var current = 84;
    var minVisible = new Promise(function (resolve) { window.setTimeout(resolve, 1150); });
    var creep = window.setInterval(function () {
      if (!loader.isOpen) {
        window.clearInterval(creep);
        return;
      }
      current += Math.max(0.05, (95 - current) * 0.045);
      if (current > 95) { current = 95; }
      loader.setProgress(current, "正在穿过最后一片星星");
    }, 180);

    var imageReady = waitForImage(4200);

    Promise.all([minVisible, imageReady]).then(function () {
      window.clearInterval(creep);
      loader.setProgress(97, "就快到了");
      window.setTimeout(function () {
        loader.finish();
        window.setTimeout(function () { loader.hide(); }, 820);
      }, 220);
    });
  }

  onReady(function () {
    // 上一次上传失败时 sessionStorage 里可能还留着 pending 标记；
    // 非结果页必须立刻清掉，否则加载遮罩会一直挡住首页。
    if (!document.querySelector(".result-card") && hasPending()) {
      clearPending();
      if (loader.isOpen) { loader.hide(); }
    }

    var uploadForm = document.querySelector("[data-postcard-form]");
    if (uploadForm) {
      var input = uploadForm.querySelector('input[type="file"][name="image"]');
      var submitting = false;

      uploadForm.addEventListener("submit", function (event) {
        if (submitting) { return; }
        if (!input || !input.files || !input.files.length) { return; }
        if (event.defaultPrevented) { return; }

        submitting = true;
        event.preventDefault();

        setPending();
        startUploadOverlay();

        // Give the overlay one paint before native form navigation.
        window.requestAnimationFrame(function () {
          window.setTimeout(function () {
            // Bypass the submit event, but keep the form's native multipart body.
            HTMLFormElement.prototype.submit.call(uploadForm);
          }, 140);
        });
      });
    }

    if (document.querySelector(".result-card") && (hasPending() || document.documentElement.classList.contains(PENDING_CLASS))) {
      startResultOverlay();
    }
  });
})();
