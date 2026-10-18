"""路由与安全测试（计划书 §17.3、§17.4）。

使用降级模式（MODEL_DOWNGRADE）创建应用，避免测试依赖真实 CLIP 模型；
匹配逻辑本身已在 test_matcher.py 中覆盖。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import db  # noqa: E402
from tests.conftest import make_image_bytes  # noqa: E402


def _isolate_config(monkeypatch, tmp_path):
    """把 config 中所有会写盘的路径指向 tmp_path。

    注意：config 里的路径是**模块级常量**，在 import 时就已算好。
    只改 DATA_DIR 是不够的——ensure_directories() 遍历的是
    SOURCE_IMAGES_DIR / CLIP_CACHE_DIR 这些派生常量，
    它们仍指向真实项目目录，会污染开发环境（尤其是模型缓存）。
    因此必须逐个覆盖。
    """
    static_dir = tmp_path / "static"
    data_dir = tmp_path / "data"
    for sub in ("uploads", "audio", "comment_images"):
        (static_dir / sub).mkdir(parents=True, exist_ok=True)
    (data_dir / "source_images").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "STATIC_DIR", static_dir)
    monkeypatch.setattr(config, "DB_PATH", data_dir / "music.db")
    monkeypatch.setattr(config, "UPLOAD_DIR", static_dir / "uploads")
    monkeypatch.setattr(config, "AUDIO_DIR", static_dir / "audio")
    monkeypatch.setattr(config, "COMMENT_IMAGE_DIR", static_dir / "comment_images")
    monkeypatch.setattr(config, "SOURCE_IMAGES_DIR", data_dir / "source_images")
    monkeypatch.setattr(config, "CLIP_CACHE_DIR", data_dir / "model_cache")
    monkeypatch.setattr(config, "SECRET_KEY_FILE", data_dir / ".secret_key")
    monkeypatch.setattr(config, "LOG_DIR", data_dir / "logs")
    monkeypatch.setattr(config, "MODEL_DOWNGRADE", True)
    # 单元测试默认关闭访客密码门禁，单独的门禁测试再显式打开
    monkeypatch.setattr(config, "ACCESS_PASSWORD", "")
    # 开发者删除口令在代码中没有默认值，测试里显式给一个
    monkeypatch.setattr(config, "DEV_DELETE_PASSWORD", "test-dev-password")
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret-key")
    return static_dir


@pytest.fixture
def client(tmp_path, monkeypatch):
    """隔离环境：临时数据库、临时 static 目录、降级模式。"""
    _isolate_config(monkeypatch, tmp_path)

    db.init_db()
    with db.get_connection() as conn:
        db.upsert_song(
            conn,
            song_key=config.FALLBACK_SONG_KEY,
            song_name="兜底歌曲",
            artist="测试歌手",
        )

    import app as app_module

    application = app_module.create_app()
    application.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    with application.test_client() as test_client:
        yield test_client


class TestBasicRoutes:
    def test_index_ok(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "上传" in response.get_data(as_text=True)

    def test_posts_page_ok(self, client):
        response = client.get("/posts")
        assert response.status_code == 200

    def test_posts_search_by_song(self, client):
        with db.get_connection() as conn:
            song = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
            db.insert_post(
                conn,
                song_id=int(song["id"]),
                upload_image_path="uploads/search.jpg",
                nickname="甲",
                content="搜索测试",
            )
        body = client.get("/posts?q=兜底").get_data(as_text=True)
        assert "找到 1 张明信片" in body
        assert "搜索测试" in body

    def test_unknown_song_404(self, client):
        assert client.get("/song/99999").status_code == 404

    def test_unknown_postcard_404(self, client):
        assert client.get("/postcard/99999").status_code == 404

    def test_upload_without_file_redirects(self, client):
        response = client.post("/upload", data={}, follow_redirects=True)
        assert response.status_code == 200
        assert "还没有照片" in response.get_data(as_text=True)

    def test_result_without_session_redirects(self, client):
        response = client.get("/result", follow_redirects=True)
        assert response.status_code == 200
        assert "还没有寄出照片" in response.get_data(as_text=True)


class TestUploadValidation:
    def test_valid_jpeg_upload_produces_result(self, client):
        data = {"image": (io.BytesIO(make_image_bytes(fmt="JPEG")), "photo.jpg")}
        response = client.post(
            "/upload", data=data, content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "星空的回信" in body
        # 降级模式明确标注为默认歌曲
        assert "先为你留下一首默认歌曲" in body

    def test_invalid_file_rejected(self, client):
        data = {"image": (io.BytesIO(b"not an image at all"), "fake.jpg")}
        response = client.post(
            "/upload", data=data, content_type="multipart/form-data",
            follow_redirects=True,
        )
        body = response.get_data(as_text=True)
        assert "星空的回信" not in body  # 不应进入结果页

    def test_no_score_shown_on_result_page(self, client):
        data = {"image": (io.BytesIO(make_image_bytes()), "photo.jpg")}
        response = client.post(
            "/upload", data=data, content_type="multipart/form-data",
            follow_redirects=True,
        )
        body = response.get_data(as_text=True)
        for forbidden in ("相似度", "百分比", "置信度", "评分"):
            assert forbidden not in body


class TestPostSubmission:
    def _upload(self, client):
        data = {"image": (io.BytesIO(make_image_bytes()), "photo.jpg")}
        return client.post(
            "/upload", data=data, content_type="multipart/form-data",
            follow_redirects=True,
        )

    def _get_token(self, client):
        with client.session_transaction() as sess:
            return sess.get("result", {}).get("token")

    def test_post_success_and_appears_on_wall(self, client):
        self._upload(client)
        token = self._get_token(client)
        assert token

        response = client.post(
            "/submit_post",
            data={"token": token, "nickname": "小明", "content": "很好听"},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert "很好听" in response.get_data(as_text=True)

    def test_post_creates_one_matchable_note(self, client):
        """用户最终留下时只写一张可参与检索的图片笔记，不再写 post 副本。"""
        self._upload(client)
        token = self._get_token(client)
        content = "最终留下并参与匹配的明信片"
        client.post(
            "/submit_post",
            data={"token": token, "nickname": "A", "content": content},
            follow_redirects=True,
        )

        with db.get_connection() as conn:
            assert conn.execute(
                "SELECT 1 FROM posts WHERE content = ?", (content,)
            ).fetchone() is None
            note = conn.execute(
                "SELECT comment_image_id FROM comment_image_notes WHERE note_text = ?",
                (content,),
            ).fetchone()
            assert note is not None
            assert conn.execute(
                "SELECT 1 FROM song_comment_images WHERE id = ?",
                (note["comment_image_id"],),
            ).fetchone() is not None
            assert conn.execute(
                "SELECT 1 FROM item_texts "
                "WHERE item_type = 'comment_image' AND item_id = ?",
                (note["comment_image_id"],),
            ).fetchone() is not None

        wall = client.get("/posts").get_data(as_text=True)
        assert wall.count(content) == 1

    def test_empty_nickname_becomes_anonymous(self, client):
        self._upload(client)
        token = self._get_token(client)
        client.post(
            "/submit_post",
            data={"token": token, "nickname": "", "content": "内容"},
            follow_redirects=True,
        )
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT note_author FROM comment_image_notes WHERE note_text = ?",
                ("内容",),
            ).fetchone()
        assert row is not None
        assert row["note_author"] == config.DEFAULT_NICKNAME

    def test_content_length_enforced(self, client):
        self._upload(client)
        token = self._get_token(client)
        long_content = "x" * (config.POST_CONTENT_MAX_LENGTH + 1)
        response = client.post(
            "/submit_post",
            data={"token": token, "nickname": "A", "content": long_content},
            follow_redirects=True,
        )
        assert "想留下的话" in response.get_data(as_text=True)

    def test_wrong_token_rejected(self, client):
        self._upload(client)
        response = client.post(
            "/submit_post",
            data={"token": "forged-token", "nickname": "A", "content": "内容"},
            follow_redirects=True,
        )
        assert "这次寄信已经过期" in response.get_data(as_text=True)
        with db.get_connection() as conn:
            assert db.count_posts(conn) == 0

    def test_duplicate_submission_blocked(self, client):
        self._upload(client)
        token = self._get_token(client)
        payload = {"token": token, "nickname": "A", "content": "第一次"}

        client.post("/submit_post", data=payload, follow_redirects=True)
        response = client.post("/submit_post", data=payload, follow_redirects=True)

        with db.get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM comment_image_notes WHERE note_text = ?",
                ("第一次",),
            ).fetchone()[0]
            assert count == 1  # 只产生一张明信片
        assert "已经留在星空" in response.get_data(as_text=True)

    def test_wall_links_song_to_space_and_card_to_postcard(self, client):
        """歌名/歌手进歌曲空间；图片和「展开明信片」进明信片。"""
        self._upload(client)
        token = self._get_token(client)
        content = "今晚的星星"
        client.post(
            "/submit_post",
            data={"token": token, "nickname": "A", "content": content},
            follow_redirects=True,
        )

        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT ci.id, ci.song_id "
                "FROM song_comment_images AS ci "
                "JOIN comment_image_notes AS cin ON cin.comment_image_id = ci.id "
                "WHERE cin.note_text = ? ORDER BY ci.id DESC LIMIT 1",
                (content,),
            ).fetchone()
        assert row is not None

        wall = client.get("/posts").get_data(as_text=True)
        assert f"/postcard/note/{row['id']}" in wall
        assert f"/song/{row['song_id']}" in wall

        detail = client.get(f"/postcard/note/{row['id']}")
        assert detail.status_code == 200
        body = detail.get_data(as_text=True)
        assert "一张留在星空的明信片" in body
        assert content in body
        assert "走进这首歌的星群" in body
        assert f"/song/{row['song_id']}" in body

    def test_forged_song_id_ignored(self, client):
        """修改隐藏字段不能给任意歌曲发帖。"""
        self._upload(client)
        token = self._get_token(client)

        with db.get_connection() as conn:
            other_id, _ = db.upsert_song(conn, song_key="demo_other", song_name="别的歌")

        client.post(
            "/submit_post",
            data={
                "token": token,
                "nickname": "A",
                "content": "内容",
                "song_id": str(other_id),           # 伪造
                "upload_image_path": "uploads/evil.jpg",  # 伪造
            },
            follow_redirects=True,
        )

        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT ci.song_id, ci.image_local_path "
                "FROM song_comment_images AS ci "
                "JOIN comment_image_notes AS cin ON cin.comment_image_id = ci.id "
                "WHERE cin.note_text = ?",
                ("内容",),
            ).fetchall()

        assert len(rows) == 1
        assert rows[0]["song_id"] != other_id            # 以会话为准
        assert rows[0]["image_local_path"] != "uploads/evil.jpg"

    def test_html_not_interpreted(self, client):
        """留言内容正确转义，不解析 HTML。"""
        self._upload(client)
        token = self._get_token(client)
        response = client.post(
            "/submit_post",
            data={"token": token, "nickname": "A",
                  "content": "<script>alert(1)</script>"},
            follow_redirects=True,
        )
        body = response.get_data(as_text=True)
        assert "<script>alert(1)</script>" not in body
        assert "&lt;script&gt;" in body


class TestHybridRoutes:
    """单流程：图片 + 可选留言 → 图片笔记组合（降级模式下不依赖真实模型）。"""

    def test_index_has_message_and_no_target(self, client):
        body = client.get("/").get_data(as_text=True)
        assert 'name="message"' in body
        assert 'name="nickname"' in body
        assert 'name="target"' not in body

    def test_message_too_long_rejected(self, client):
        data = {
            "message": "x" * (config.QUERY_TEXT_MAX_LENGTH + 1),
            "image": (io.BytesIO(make_image_bytes()), "photo.jpg"),
        }
        response = client.post(
            "/upload", data=data, content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert "想写下的话最多" in response.get_data(as_text=True)

    def test_song_space_page_ok(self, client):
        with db.get_connection() as conn:
            row = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
        response = client.get(f"/song/{row['id']}")
        body = response.get_data(as_text=True)
        assert response.status_code == 200
        assert "这首歌下的明信片" in body


class TestSongSpace:
    """歌曲空间：笔记流、共鸣之心、留言回复（降级模式下不依赖模型）。"""

    def _seed_note(self, author=None, text="测试笔记正文"):
        import uuid

        with db.get_connection() as conn:
            song = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
            image_id, _ = db.upsert_comment_image(
                conn,
                song_id=song["id"],
                image_local_path=f"comment_images/test/{uuid.uuid4().hex}.jpg",
                source_url=None,
                image_sha256=uuid.uuid4().hex * 2,
                embedding=[0.0] * config.EMBEDDING_DIM,
                embedding_model=config.CLIP_MODEL_NAME,
            )
            db.upsert_comment_note(
                conn,
                comment_image_id=image_id,
                note_text=text,
                note_author=author,
            )
        return int(song["id"]), image_id

    def test_space_shows_note_feed_with_default_author(self, client):
        song_id, _image_id = self._seed_note(author=None)
        body = client.get(f"/song/{song_id}").get_data(as_text=True)
        assert "测试笔记正文" in body
        assert "共鸣之心" in body
        assert "风外客" in body

    def test_like_toggles_on_and_off(self, client):
        song_id, image_id = self._seed_note()

        response = client.post(f"/note/{image_id}/like", follow_redirects=True)
        assert response.status_code == 200
        assert "已共鸣" in response.get_data(as_text=True)
        with db.get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM note_likes WHERE comment_image_id = ?",
                (image_id,),
            ).fetchone()[0]
        assert total == 1

        response = client.post(f"/note/{image_id}/like", follow_redirects=True)
        assert "已取消共鸣" in response.get_data(as_text=True)
        with db.get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM note_likes WHERE comment_image_id = ?",
                (image_id,),
            ).fetchone()[0]
        assert total == 0

        client.post(f"/note/{image_id}/like", follow_redirects=True)
        with db.get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM note_likes WHERE comment_image_id = ?",
                (image_id,),
            ).fetchone()[0]
        assert total == 1

    def test_note_card_opens_a_postcard_page(self, client):
        """素材笔记（kind=note）也要能展开成明信片，而不是只能去歌曲空间。"""
        song_id, image_id = self._seed_note(text="这张笔记也要成为明信片")
        detail = client.get(f"/postcard/note/{image_id}")
        assert detail.status_code == 200
        body = detail.get_data(as_text=True)
        assert "一张留在星空的明信片" in body
        assert "这张笔记也要成为明信片" in body
        assert "走进这首歌的星群" in body
        assert f"/song/{song_id}" in body

    def test_postcard_interactions_are_shared_with_song_space(self, client):
        """明信片页的共鸣和回信，与歌曲空间是同一份数据。"""
        song_id, image_id = self._seed_note(text="共享共鸣的明信片")

        client.post(f"/note/{image_id}/like", follow_redirects=True)
        client.post(
            f"/note/{image_id}/reply",
            data={"content": "在明信片背面回信"},
            follow_redirects=True,
        )

        body = client.get(f"/postcard/note/{image_id}").get_data(as_text=True)
        assert "已共鸣" in body
        assert "在明信片背面回信" in body
        assert f"/note/{image_id}/like" in body
        assert f"/note/{image_id}/reply" in body

        song_body = client.get(f"/song/{song_id}").get_data(as_text=True)
        assert "已共鸣" in song_body
        assert "在明信片背面回信" in song_body

    def test_new_note_joins_matching_index_without_restart(self, client):
        """新写入的图片笔记刷新索引后，下一张照片可以命中它。"""
        import uuid

        import numpy as np

        image_path = config.COMMENT_IMAGE_DIR / "dynamic.jpg"
        image_path.write_bytes(make_image_bytes())
        vector = [1.0] + [0.0] * (config.EMBEDDING_DIM - 1)

        with db.get_connection() as conn:
            song = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
            image_id, _ = db.upsert_comment_image(
                conn,
                song_id=int(song["id"]),
                image_local_path="comment_images/dynamic.jpg",
                source_url=None,
                image_sha256=uuid.uuid4().hex * 2,
                embedding=vector,
                embedding_model=config.CLIP_MODEL_NAME,
            )
            db.upsert_comment_note(
                conn,
                comment_image_id=image_id,
                note_text="刚刚加入星群的新明信片",
                note_author="A",
            )

        service = client.application.extensions["hybrid_service"]
        before = service.comment_index_size
        service.refresh_comment_index()
        assert service.comment_index_size == before + 1

        outcome = service.match_song(np.asarray(vector, dtype=np.float32))
        assert outcome is not None
        assert outcome.item.item_id == image_id

    def test_post_card_interactions_shared_with_song_space(self, client):
        """用户投递的 post 明信片，在明信片页与歌曲空间共享共鸣/回信。"""
        with db.get_connection() as conn:
            song = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
            post_id = db.insert_post(
                conn,
                song_id=int(song["id"]),
                upload_image_path="uploads/test-post.jpg",
                nickname="A",
                content="来自星空的一张用户明信片",
            )

        detail = client.get(f"/postcard/{post_id}")
        assert detail.status_code == 200
        body = detail.get_data(as_text=True)
        assert "共鸣之心" in body
        assert "回信" in body

        client.post(f"/postcard/{post_id}/like", follow_redirects=True)
        client.post(
            f"/postcard/{post_id}/reply",
            data={
                "content": "这张明信片的回信",
                "next": f"/postcard/{post_id}",
            },
            follow_redirects=True,
        )

        body = client.get(f"/postcard/{post_id}").get_data(as_text=True)
        assert "已共鸣" in body
        assert "这张明信片的回信" in body

        song_body = client.get(f"/song/{song['id']}").get_data(as_text=True)
        assert "这首歌下的明信片" in song_body
        assert "从星空寄来的明信片" not in song_body
        assert "来自星空的一张用户明信片" in song_body
        assert "这张明信片的回信" in song_body

    def test_duplicate_note_and_post_only_show_once(self, client):
        """素材 note 与用户 post 内容、作者、日期相同时，星群只展示一张。"""
        import time
        import uuid

        content = "同一张明信片不要出现两遍"
        with db.get_connection() as conn:
            song = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
            db.insert_post(
                conn,
                song_id=int(song["id"]),
                upload_image_path="uploads/duplicate-post.jpg",
                nickname="A",
                content=content,
            )
            image_id, _ = db.upsert_comment_image(
                conn,
                song_id=int(song["id"]),
                image_local_path=f"comment_images/test/{uuid.uuid4().hex}.jpg",
                source_url=None,
                image_sha256=uuid.uuid4().hex * 2,
                embedding=[0.0] * config.EMBEDDING_DIM,
                embedding_model=config.CLIP_MODEL_NAME,
            )
            db.upsert_comment_note(
                conn,
                comment_image_id=image_id,
                note_text=content,
                note_author="A",
                note_time_ms=int(time.time() * 1000),
            )

        body = client.get(f"/song/{song['id']}").get_data(as_text=True)
        assert body.count(content) == 1

    def test_leave_postcard_on_song_page(self, client):
        """播放器下面可以直接留一张绑定当前歌曲的明信片。"""
        with db.get_connection() as conn:
            song = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
        assert song is not None

        response = client.post(
            f"/song/{song['id']}/postcard",
            data={
                "image": (io.BytesIO(make_image_bytes()), "star.jpg"),
                "content": "留在播放器下面的明信片",
                "nickname": "A",
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert "留在播放器下面的明信片" in response.get_data(as_text=True)

        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT comment_image_id, note_text FROM comment_image_notes WHERE note_text = ?",
                ("留在播放器下面的明信片",),
            ).fetchone()
            assert row is not None
            assert conn.execute(
                "SELECT 1 FROM song_comment_images WHERE id = ?",
                (row["comment_image_id"],),
            ).fetchone() is not None
            assert conn.execute(
                "SELECT 1 FROM item_texts "
                "WHERE item_type = 'comment_image' AND item_id = ?",
                (row["comment_image_id"],),
            ).fetchone() is not None

    def test_like_ajax_returns_state_without_reloading(self, client):
        """共鸣用 JSON 返回，页面不跳转，歌曲可以继续播放。"""
        song_id, image_id = self._seed_note()

        response = client.post(
            f"/note/{image_id}/like",
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data == {"liked": True, "likes": 1}

        response = client.post(
            f"/note/{image_id}/like",
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data == {"liked": False, "likes": 0}

    def test_reply_defaults_to_guest_and_is_visible(self, client):
        song_id, image_id = self._seed_note()
        response = client.post(
            f"/note/{image_id}/reply",
            data={"nickname": "", "content": "这条回复不署名"},
            follow_redirects=True,
        )
        body = response.get_data(as_text=True)
        assert "回信已经放进星群" in body
        assert "这条回复不署名" in body
        assert "风外客" in body

    def test_reply_ignores_supplied_nickname(self, client):
        song_id, image_id = self._seed_note()
        client.post(
            f"/note/{image_id}/reply",
            data={"nickname": "小明", "content": "统一匿名的回复"},
            follow_redirects=True,
        )
        body = client.get(f"/song/{song_id}").get_data(as_text=True)
        assert "统一匿名的回复" in body
        assert "小明" not in body

    def test_reply_rejects_empty_content(self, client):
        song_id, image_id = self._seed_note()
        response = client.post(
            f"/note/{image_id}/reply",
            data={"nickname": "", "content": "   "},
            follow_redirects=True,
        )
        assert "回信不能为空" in response.get_data(as_text=True)
        with db.get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM note_replies WHERE comment_image_id = ?",
                (image_id,),
            ).fetchone()[0]
        assert total == 0

    def test_like_unknown_note_404(self, client):
        assert client.post("/note/999999/like").status_code == 404


class TestCsrf:
    def test_csrf_protection_active(self, tmp_path, monkeypatch):
        """启用 CSRF 时，无 token 的 POST 应被拒绝。"""
        _isolate_config(monkeypatch, tmp_path)

        db.init_db()
        with db.get_connection() as conn:
            db.upsert_song(conn, song_key=config.FALLBACK_SONG_KEY, song_name="兜底")

        import app as app_module

        application = app_module.create_app()
        application.config.update(TESTING=True)  # 保持 CSRF 开启

        with application.test_client() as test_client:
            data = {"image": (io.BytesIO(make_image_bytes()), "photo.jpg")}
            response = test_client.post(
                "/upload", data=data, content_type="multipart/form-data"
            )
            assert response.status_code == 400  # CSRF 拒绝


class TestUnavailablePaths:
    """计划书 §14：数据库不可访问 / 兜底歌曲不存在 都必须返回 503。

    这两个分支此前完全没有测试覆盖，而它们的正确性依赖
    路由上**装饰器的先后顺序**——一旦顺序被改，用户会看到
    指向错误方向的提示（例如数据库坏了却提示"兜底歌曲未配置"）。
    """

    def _make_app(self, tmp_path, monkeypatch, *, db_ok: bool, fallback_ok: bool):
        _isolate_config(monkeypatch, tmp_path)

        db.init_db()
        with db.get_connection() as conn:
            db.upsert_song(conn, song_key=config.FALLBACK_SONG_KEY, song_name="兜底")

        import app as app_module

        monkeypatch.setattr(app_module.db, "check_database", lambda: db_ok)
        monkeypatch.setattr(app_module, "_check_fallback_song", lambda: fallback_ok)

        application = app_module.create_app()
        application.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
        return application

    def test_database_unavailable_returns_503(self, tmp_path, monkeypatch):
        application = self._make_app(tmp_path, monkeypatch,
                                     db_ok=False, fallback_ok=True)
        with application.test_client() as client:
            response = client.get("/")
            assert response.status_code == 503

    def test_fallback_missing_returns_503_on_upload(self, tmp_path, monkeypatch):
        application = self._make_app(tmp_path, monkeypatch,
                                     db_ok=True, fallback_ok=False)
        with application.test_client() as client:
            data = {"image": (io.BytesIO(make_image_bytes()), "photo.jpg")}
            response = client.post("/upload", data=data,
                                   content_type="multipart/form-data")
            assert response.status_code == 503

    def test_database_error_message_wins_over_fallback(self, tmp_path, monkeypatch):
        """数据库与兜底同时异常时，必须报"数据库"而不是"兜底歌曲"。

        这锁定了 @require_database 位于 @require_fallback 外层这一事实。

        注意：必须用同时挂两个装饰器的路由（/upload、/submit_post）。
        "/" 上只有 @require_database，用它做断言无法区分顺序——
        那样即使装饰器被调换，测试也会通过（假的安心）。
        """
        application = self._make_app(tmp_path, monkeypatch,
                                     db_ok=False, fallback_ok=False)
        with application.test_client() as client:
            data = {"image": (io.BytesIO(make_image_bytes()), "photo.jpg")}
            response = client.post("/upload", data=data,
                                   content_type="multipart/form-data")
            body = response.get_data(as_text=True)
            assert response.status_code == 503
            assert "数据库" in body
            assert "兜底" not in body

    def test_app_starts_without_database(self, tmp_path, monkeypatch):
        """计划书 §14 末条：数据库缺失时可以启动并给出友好提示，不得崩溃。"""
        application = self._make_app(tmp_path, monkeypatch,
                                     db_ok=False, fallback_ok=False)
        assert application is not None
        with application.test_client() as client:
            # 不抛异常、不是 500
            assert client.get("/").status_code == 503


class TestExplicitPublish:
    """检索阶段只算匹配，不写库；用户明确点「发布」才产生留言。"""

    def _upload(self, client, message="今天想听点旧的", nickname="小明"):
        data = {
            "message": message,
            "nickname": nickname,
            "image": (io.BytesIO(make_image_bytes()), "photo.jpg"),
        }
        return client.post(
            "/upload", data=data, content_type="multipart/form-data",
            follow_redirects=True,
        )

    def test_upload_with_message_does_not_write(self, client):
        response = self._upload(client)
        assert response.status_code == 200
        body = response.get_data(as_text=True)

        assert "留在星空" in body              # 结果页给出留下按钮
        assert "今天想听点旧的" in body            # 输入被带到结果页

        with db.get_connection() as conn:
            assert db.count_posts(conn) == 0        # 还没落库

        with client.session_transaction() as sess:
            stored = sess["result"]
            assert stored["post_used"] is False
            assert stored["query_message"] == "今天想听点旧的"

    def test_publish_then_writes_exactly_one(self, client):
        self._upload(client)
        with client.session_transaction() as sess:
            token = sess["result"]["token"]

        first = client.post(
            "/submit_post",
            data={"token": token, "nickname": "小明", "content": "今天想听点旧的"},
            follow_redirects=True,
        )
        assert "你的明信片已经留在星空" in first.get_data(as_text=True)

        # 同一个结果再发一次也不会多出一条
        client.post(
            "/submit_post",
            data={"token": token, "nickname": "小明", "content": "今天想听点旧的"},
            follow_redirects=True,
        )
        with db.get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM comment_image_notes WHERE note_text = ?",
                ("今天想听点旧的",),
            ).fetchone()[0]
            assert count == 1


class TestDevDelete:
    """留言墙的开发者入口：密码通过后可以删除帖子。"""

    def _seed_post(self):
        with db.get_connection() as conn:
            song = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
            post_id = db.insert_post(
                conn,
                song_id=int(song["id"]),
                upload_image_path="uploads/test.jpg",
                nickname="测试",
                content="这条要被删掉",
            )
        return int(post_id)

    def test_wall_has_dev_entry_but_no_delete_buttons(self, client):
        self._seed_post()
        body = client.get("/posts").get_data(as_text=True)
        assert "开发者" in body
        assert "dev-delete-btn" not in body

    def test_wrong_password_rejected(self, client):
        response = client.post(
            "/dev/login", data={"password": "000000"}, follow_redirects=True
        )
        assert "密码不正确" in response.get_data(as_text=True)
        with client.session_transaction() as sess:
            assert not sess.get("dev_mode")

    def test_login_shows_delete_buttons(self, client):
        self._seed_post()
        response = client.post(
            "/dev/login",
            data={"password": config.DEV_DELETE_PASSWORD},
            follow_redirects=True,
        )
        body = response.get_data(as_text=True)
        assert "开发者模式" in body
        assert "dev-delete-btn" in body
        with client.session_transaction() as sess:
            assert sess.get("dev_mode") is True

    def test_delete_removes_post(self, client):
        post_id = self._seed_post()
        client.post(
            "/dev/login",
            data={"password": config.DEV_DELETE_PASSWORD},
            follow_redirects=True,
        )
        response = client.post(
            f"/dev/post/{post_id}/delete", data={"page": "1"},
            follow_redirects=True,
        )
        assert "已删除明信片" in response.get_data(as_text=True)
        with db.get_connection() as conn:
            assert db.count_posts(conn) == 0
            assert db.get_post(conn, post_id) is None

    def test_delete_requires_dev_mode(self, client):
        post_id = self._seed_post()
        response = client.post(
            f"/dev/post/{post_id}/delete", follow_redirects=True
        )
        assert "需要先进入开发者模式" in response.get_data(as_text=True)
        with db.get_connection() as conn:
            assert db.count_posts(conn) == 1

    def test_logout_disables_delete(self, client):
        post_id = self._seed_post()
        client.post(
            "/dev/login",
            data={"password": config.DEV_DELETE_PASSWORD},
            follow_redirects=True,
        )
        response = client.post("/dev/logout", follow_redirects=True)
        assert "已退出开发者模式" in response.get_data(as_text=True)

        blocked = client.post(
            f"/dev/post/{post_id}/delete", follow_redirects=True
        )
        assert "需要先进入开发者模式" in blocked.get_data(as_text=True)
        with db.get_connection() as conn:
            assert db.count_posts(conn) == 1

    def test_delete_does_not_rely_on_confirm_dialog(self, client):
        """浏览器可能拦掉 confirm()，被拦时它返回 false，表单就永远不提交。"""
        self._seed_post()
        client.post(
            "/dev/login",
            data={"password": config.DEV_DELETE_PASSWORD},
            follow_redirects=True,
        )
        body = client.get("/posts").get_data(as_text=True)
        assert "confirm(" not in body
        assert "data-confirm-delete" in body

    def test_html_is_not_cached(self, client):
        """删除后刷新必须看到最新列表，不能被浏览器缓存挡住。"""
        response = client.get("/posts")
        assert "no-store" in response.headers.get("Cache-Control", "")
