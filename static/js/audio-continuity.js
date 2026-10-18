/* ============================================================
   歌曲进度接力
   ------------------------------------------------------------
   在「星空的回信」或单张明信片里点「走进这首歌的星群」时，
   把当前播放位置交给歌曲页，避免同名歌曲从 0:00 重新开始。
   ============================================================ */
(function () {
  'use strict';

  var KEY = 'lpAudioResume';
  var MAX_AGE = 6 * 60 * 60 * 1000;
  var audio = document.getElementById('player');
  if (!audio) { return; }

  /* 离开明信片前，记住这首歌播到了哪里。 */
  var links = document.querySelectorAll('a[href*="/song/"]');
  for (var i = 0; i < links.length; i++) {
    links[i].addEventListener('click', function () {
      if (!audio || audio.currentTime <= 0.25) { return; }
      try {
        sessionStorage.setItem(KEY, JSON.stringify({
          songId: audio.getAttribute('data-song-id') || '',
          time: audio.currentTime,
          paused: audio.paused,
          savedAt: Date.now()
        }));
      } catch (error) {}
    });
  }

  /* 只有歌曲页才恢复进度；明信片页只负责写入。 */
  if (document.body.getAttribute('data-page') !== 'song_detail') { return; }
  var pageSongId = audio.getAttribute('data-song-id');
  if (!pageSongId) { return; }

  var stored = null;
  try {
    var raw = sessionStorage.getItem(KEY);
    if (raw) { stored = JSON.parse(raw); }
    sessionStorage.removeItem(KEY);
  } catch (error) {
    return;
  }

  if (!stored || !stored.savedAt || Date.now() - stored.savedAt > MAX_AGE) { return; }
  if (String(stored.songId) !== String(pageSongId)) { return; }

  var hint = document.getElementById('audio-resume-hint');

  function showHint(text) {
    if (!hint) { return; }
    hint.textContent = text;
    hint.hidden = false;
  }

  function applyProgress() {
    var target = Number(stored.time) || 0;
    if (target > 0.25) {
      try {
        var max = (isFinite(audio.duration) && audio.duration > 0)
          ? Math.max(0, audio.duration - 0.25)
          : target;
        audio.currentTime = Math.min(target, max);
      } catch (error) {}
    }

    if (stored.paused) {
      showHint('已回到上次暂停的位置。');
      return;
    }

    var attempt = audio.play();
    if (attempt && typeof attempt.then === 'function') {
      attempt.catch(function () {
        showHint('播放进度已保留，点一下播放键继续。');
      });
    }
  }

  if (audio.readyState >= 1) {
    applyProgress();
  } else {
    audio.preload = 'auto';
    audio.addEventListener('loadedmetadata', applyProgress, { once: true });
    audio.load();
  }
}());
