"""访客访问密码门禁测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import db  # noqa: E402
from tests.test_routes import _isolate_config  # noqa: E402


@pytest.fixture
def gated_client(tmp_path, monkeypatch):
    _isolate_config(monkeypatch, tmp_path)
    # 测试用假口令：与真实部署无关，代码中没有任何默认口令
    monkeypatch.setattr(config, "ACCESS_PASSWORD", "test-access-pass")
    db.init_db()
    with db.get_connection() as conn:
        db.upsert_song(
            conn, song_key=config.FALLBACK_SONG_KEY, song_name="兜底歌曲"
        )

    import app as app_module

    application = app_module.create_app()
    application.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    with application.test_client() as test_client:
        yield test_client


def test_root_requires_password(gated_client):
    response = gated_client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]

    login_page = gated_client.get("/login")
    assert login_page.status_code == 200
    assert "访问密码" in login_page.get_data(as_text=True)


def test_media_blocked_but_css_allowed(gated_client):
    css = gated_client.get("/static/css/style.css", follow_redirects=False)
    assert css.status_code != 302

    audio = gated_client.get("/static/audio/x.mp3", follow_redirects=False)
    assert audio.status_code == 302
    assert "/login" in audio.headers["Location"]


def test_wrong_password_stays_on_login(gated_client):
    response = gated_client.post(
        "/login", data={"password": "000000"}, follow_redirects=True
    )
    assert "访问密码不正确" in response.get_data(as_text=True)


def test_correct_password_opens_site(gated_client):
    response = gated_client.post(
        "/login", data={"password": "test-access-pass", "next": "/posts"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "明信片的星空" in response.get_data(as_text=True)

    home = gated_client.get("/", follow_redirects=False)
    assert home.status_code == 200


def test_logout_locks_again(gated_client):
    gated_client.post("/login", data={"password": "test-access-pass"})
    gated_client.get("/logout")
    response = gated_client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
