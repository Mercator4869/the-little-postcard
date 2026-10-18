"""星空墙搜索：按歌名 / 歌手过滤（纯数据库逻辑）。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import db  # noqa: E402


def _seed(conn):
    rain, _ = db.upsert_song(
        conn, song_key="test_rain", song_name="雨爱", artist="杨丞琳"
    )
    stubborn, _ = db.upsert_song(
        conn, song_key="test_stubborn", song_name="倔强", artist="五月天"
    )
    db.insert_post(
        conn,
        song_id=rain,
        upload_image_path="uploads/a.jpg",
        nickname="甲",
        content="雨天",
    )
    db.insert_post(
        conn,
        song_id=stubborn,
        upload_image_path="uploads/b.jpg",
        nickname="乙",
        content="倔强",
    )


def test_filter_by_song_name(temp_db):
    with db.get_connection() as conn:
        _seed(conn)
    with db.get_connection() as conn:
        assert db.count_wall_cards(conn) == 2
        assert db.count_wall_cards(conn, query="雨爱") == 1
        rows = db.list_wall_cards(conn, page=1, page_size=10, query="雨爱")
        assert [row["song_name"] for row in rows] == ["雨爱"]


def test_filter_by_artist(temp_db):
    with db.get_connection() as conn:
        _seed(conn)
    with db.get_connection() as conn:
        assert db.count_wall_cards(conn, query="五月天") == 1
        assert db.count_wall_cards(conn, query="杨丞琳") == 1
        assert db.count_wall_cards(conn, query="不存在") == 0


def test_wildcards_are_literal(temp_db):
    with db.get_connection() as conn:
        _seed(conn)
    with db.get_connection() as conn:
        assert db.count_wall_cards(conn, query="%") == 0
        assert db.count_wall_cards(conn, query="_") == 0


def test_pagination_keeps_filter(temp_db):
    with db.get_connection() as conn:
        _seed(conn)
    with db.get_connection() as conn:
        first = db.list_wall_cards(conn, page=1, page_size=1, query="雨爱")
        second = db.list_wall_cards(conn, page=2, page_size=1, query="雨爱")
        assert len(first) == 1
        assert second == []
