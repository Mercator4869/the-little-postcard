"""数据库事务语义测试。

重点验证 get_connection 的回滚行为覆盖**所有**异常类型，
而不只是 sqlite3.Error——否则非 SQL 异常逃逸时事务状态不可预期。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import db  # noqa: E402


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "tx.db")
    db.init_db()
    return tmp_path / "tx.db"


class TestCommitSemantics:
    def test_normal_exit_commits(self, temp_db):
        with db.get_connection() as conn:
            db.upsert_song(conn, song_key="t_commit", song_name="已提交")

        with db.get_connection() as conn:
            assert db.get_song_by_key(conn, "t_commit") is not None

    def test_sqlite_error_rolls_back(self, temp_db):
        with pytest.raises(sqlite3.Error):
            with db.get_connection() as conn:
                db.upsert_song(conn, song_key="t_sql", song_name="应回滚")
                conn.execute("INSERT INTO songs (song_key) VALUES (?)", ("x",))

        with db.get_connection() as conn:
            assert db.get_song_by_key(conn, "t_sql") is None

    def test_plain_exception_rolls_back(self, temp_db):
        """非 sqlite3.Error 也必须回滚（本次修复的核心）。"""
        with pytest.raises(RuntimeError):
            with db.get_connection() as conn:
                db.upsert_song(conn, song_key="t_plain", song_name="应回滚")
                raise RuntimeError("模拟文件写入失败等非 SQL 异常")

        with db.get_connection() as conn:
            assert db.get_song_by_key(conn, "t_plain") is None

    def test_keyboard_interrupt_rolls_back(self, temp_db):
        """BaseException 也要回滚（Ctrl+C 场景）。"""
        with pytest.raises(KeyboardInterrupt):
            with db.get_connection() as conn:
                db.upsert_song(conn, song_key="t_kb", song_name="应回滚")
                raise KeyboardInterrupt

        with db.get_connection() as conn:
            assert db.get_song_by_key(conn, "t_kb") is None

    def test_foreign_key_enforced(self, temp_db):
        """外键必须开启：不存在的 song_id 不得写入评论图片。"""
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                db.upsert_comment_image(
                    conn,
                    song_id=999999,
                    image_local_path="comment_images/ghost.jpg",
                    source_url=None,
                    image_sha256="deadbeef",
                    embedding=[0.0] * config.EMBEDDING_DIM,
                    embedding_model=config.CLIP_MODEL_NAME,
                )

    def test_posts_cascade_or_restrict(self, temp_db):
        """删除歌曲时，关联留言的处置必须由外键规则明确决定（不得悬挂）。"""
        with db.get_connection() as conn:
            song_id, _ = db.upsert_song(conn, song_key="t_fk", song_name="歌")
            db.insert_post(
                conn,
                song_id=song_id,
                upload_image_path="uploads/x.jpg",
                nickname="n",
                content="内容内容内容",
            )

        # 直接删歌应被外键拒绝（或级联），总之不能留下悬挂留言
        with db.get_connection() as conn:
            try:
                conn.execute("DELETE FROM songs WHERE id = ?", (song_id,))
                conn.execute("SELECT 1")
            except sqlite3.IntegrityError:
                pass  # 被外键拒绝，符合预期

        with db.get_connection() as conn:
            rows = db.list_posts_paginated(conn, page=1, page_size=10)
            for row in rows:
                assert db.song_exists(conn, int(row["song_id"])), "存在悬挂留言"


class TestConnectionFailure:
    def test_unavailable_database_raises_typed_error(self, tmp_path, monkeypatch):
        """数据库不可用时必须抛 DatabaseUnavailable，供上层转 503。"""
        # 指向一个不可能的路径（把文件当目录用）
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(config, "DB_PATH", blocker / "sub" / "music.db")

        with pytest.raises(db.DatabaseUnavailable):
            with db.get_connection() as conn:
                conn.execute("SELECT 1")
