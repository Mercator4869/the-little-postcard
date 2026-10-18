/* =====================================================================
   留言墙：开发者删除的两步确认
   ----------------------------------------------------------------
   不用 window.confirm()：有些浏览器/设置会直接拦掉弹窗，confirm 返回
   false，表单就不会提交，看起来像「点了删除没反应」。
   这里改成纯 DOM 的两步确认——第一次点击把按钮变成「确认删除」，
   3.5 秒内再点一次才真正提交。
   ===================================================================== */
(function () {
  'use strict';

  var buttons = document.querySelectorAll('[data-confirm-delete]');
  if (!buttons.length) { return; }

  var armed = null;
  var timer = 0;

  function disarm() {
    if (!armed) { return; }
    armed.classList.remove('is-armed');
    armed.textContent = '删除';
    armed = null;
    window.clearTimeout(timer);
  }

  function arm(button) {
    disarm();
    armed = button;
    button.classList.add('is-armed');
    button.textContent = '确认删除';
    timer = window.setTimeout(disarm, 3500);
  }

  for (var i = 0; i < buttons.length; i++) {
    buttons[i].addEventListener('click', function (ev) {
      if (this === armed) { return; }   // 已进入确认态：放行，真正提交
      ev.preventDefault();
      arm(this);
    });
  }

  // 点别处就撤销确认态
  document.addEventListener('click', function (ev) {
    if (!armed) { return; }
    if (armed === ev.target) { return; }
    disarm();
  });
}());

/* 整张帖子都可以点开明信片；表单、链接和文字选择保持原有行为。 */
(function () {
  'use strict';

  var items = document.querySelectorAll('.post-item');
  for (var i = 0; i < items.length; i++) {
    (function (item) {
      var link = item.querySelector('.postcard-open a') ||
                 item.querySelector('.post-song a');
      if (!link) { return; }

      item.setAttribute('data-postcard-href', link.getAttribute('href'));

      item.addEventListener('click', function (ev) {
        if (ev.defaultPrevented) { return; }
        var target = ev.target;
        if (target.closest && target.closest('a,button,form,input,textarea,select,label,summary,details')) {
          return;
        }
        var selected = window.getSelection ? String(window.getSelection()) : '';
        if (selected.length > 0) { return; }
        window.location.href = link.getAttribute('href');
      });
    }(items[i]));
  }
}());
