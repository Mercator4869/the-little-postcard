/* =====================================================================
   上传页：明信片表单
   ----------------------------------------------------------------
   · 选照片 → 立刻在正面预览，并让卡片长成照片的比例（不拉伸、不裁切）
   · 没选照片就点「寄出」→ 自动翻回正面并给出提示
   翻面本身由 postcard.js 负责，这里只做照片与提交这几件事。
   ===================================================================== */
(function () {
  'use strict';

  var form = document.querySelector('[data-postcard-form]');
  var stage = document.querySelector('[data-postcard]');
  if (!form || !stage) { return; }

  var input = document.getElementById('image');
  var preview = stage.querySelector('[data-photo-preview]');
  var empty = stage.querySelector('[data-photo-empty]');
  var change = stage.querySelector('[data-photo-change]');
  var card = stage.querySelector('.postcard');
  var hint = document.querySelector('[data-postcard-hint]');
  if (!input || !preview || !card) { return; }

  var objectUrl = null;
  var HINT_EMPTY = hint ? hint.textContent : '';
  var HINT_READY = '照片放好了，翻到背面写一句话';

  function setHint(text) {
    if (hint) { hint.textContent = text; }
  }

  /* 卡片比例跟着照片走，但夹在一个表单排得下、也不会太扁的范围里 */
  function fitCardToImage() {
    if (!preview.naturalWidth || !preview.naturalHeight) { return; }
    var ratio = preview.naturalWidth / preview.naturalHeight;
    ratio = Math.max(0.9, Math.min(1.6, ratio));
    card.style.setProperty('--pc-ar', ratio.toFixed(4));
  }

  input.addEventListener('change', function () {
    var file = input.files && input.files[0];
    if (!file) { return; }

    if (objectUrl) { window.URL.revokeObjectURL(objectUrl); }
    objectUrl = window.URL.createObjectURL(file);

    preview.onload = function () {
      fitCardToImage();
      preview.hidden = false;
      if (empty) { empty.hidden = true; }
      if (change) { change.hidden = false; }
      setHint(HINT_READY);
    };
    preview.src = objectUrl;
  });

  /* 没有照片就提交时，先把人带回正面，别让原生校验气泡飘在看不见的地方 */
  form.addEventListener('submit', function (ev) {
    if (input.files && input.files.length) { return; }
    ev.preventDefault();
    if (window.DSHPostcard) { window.DSHPostcard.setFlipped(false); }
    setHint('还没有照片——轻触正面，放一张此刻的照片');
    if (empty) { empty.hidden = false; }
  });

  /* 从反面翻回正面时，把提示恢复成默认文案 */
  var backButton = stage.querySelector('.pc-back-link');
  if (backButton) {
    backButton.addEventListener('click', function () {
      setHint(input.files && input.files.length ? HINT_READY : HINT_EMPTY);
    });
  }
}());
