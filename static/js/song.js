/* ============================================================
   星群里的共鸣：只更新按钮，不整页刷新。
   ------------------------------------------------------------
   歌曲详情页的播放器在页面顶部，原来的表单提交会刷新页面、把歌停掉。
   这里用 fetch 提交同一个表单（含 CSRF），让播放不中断。
   ============================================================ */
(function () {
  'use strict';

  var forms = document.querySelectorAll('form.like-form');
  var audio = document.getElementById('player');
  var playButton = document.querySelector('[data-play-song]');

  if (audio && playButton) {
    playButton.addEventListener('click', function (event) {
      event.preventDefault();
      var attempt = audio.play();
      if (attempt && typeof attempt.catch === 'function') {
        attempt.catch(function () {
          playButton.textContent = '再点一次播放';
        });
      }
    });
  }

  if (!forms.length || !window.fetch || !window.FormData || !window.Promise) {
    return;
  }

  function render(button, liked, likes) {
    button.classList.toggle('liked', liked);
    button.setAttribute('aria-pressed', liked ? 'true' : 'false');
    button.title = liked ? '再点一次取消共鸣' : '点一下，把共鸣留在星群里';
    button.textContent = (liked ? '♥ 已共鸣 · ' : '♥ 共鸣之心 · ') + likes;
  }

  for (var i = 0; i < forms.length; i++) {
    (function (form) {
      var button = form.querySelector('.like-btn');
      if (!button) { return; }

      var busy = false;

      form.addEventListener('submit', function (event) {
        if (busy) {
          event.preventDefault();
          return;
        }

        busy = true;
        event.preventDefault();

        var oldText = button.textContent;
        var oldLiked = button.classList.contains('liked');
        button.disabled = true;

        fetch(form.action, {
          method: 'POST',
          body: new FormData(form),
          credentials: 'same-origin',
          headers: {
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': 'application/json'
          }
        }).then(function (response) {
          if (!response.ok) {
            throw new Error('like request failed: ' + response.status);
          }
          return response.json();
        }).then(function (data) {
          render(button, !!data.liked, Number(data.likes) || 0);
        }).catch(function () {
          // 不跳转，也不打断歌曲；只恢复原来的按钮状态。
          button.classList.toggle('liked', oldLiked);
          button.textContent = oldText;
          button.title = '暂时没能共鸣，再试一次';
        }).then(function () {
          button.disabled = false;
          busy = false;
        });
      });
    }(forms[i]));
  }
}());
