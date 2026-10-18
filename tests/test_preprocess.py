"""预处理测试（计划书 §17.2）。

素材导入、去重、重建安全性与 CSV 校验。
特征提取部分用假编码器，避免测试依赖真实 CLIP 模型。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import db  # noqa: E402
import preprocess  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestSongsCsv:
    def test_reads_valid_seed(self, tmp_path):
        path = tmp_path / "songs.csv"
        write_csv(path, ["song_key", "song_name"], [
            {"song_key": "demo_001", "song_name": "歌曲一"},
            {"song_key": "demo-002", "song_name": "歌曲二"},
        ])
        rows = preprocess.read_songs_csv(path)
        assert len(rows) == 2
        assert rows[0].song_key == "demo_001"

    def test_missing_required_column(self, tmp_path):
        path = tmp_path / "songs.csv"
        write_csv(path, ["song_key"], [{"song_key": "demo_001"}])
        with pytest.raises(preprocess.CsvFormatError):
            preprocess.read_songs_csv(path)

    def test_duplicate_song_key_rejected(self, tmp_path):
        path = tmp_path / "songs.csv"
        write_csv(path, ["song_key", "song_name"], [
            {"song_key": "demo_001", "song_name": "A"},
            {"song_key": "demo_001", "song_name": "B"},
        ])
        with pytest.raises(preprocess.CsvFormatError):
            preprocess.read_songs_csv(path)

    def test_invalid_song_key_rejected(self, tmp_path):
        path = tmp_path / "songs.csv"
        write_csv(path, ["song_key", "song_name"], [
            {"song_key": "../etc/passwd", "song_name": "X"},
        ])
        with pytest.raises(preprocess.CsvFormatError):
            preprocess.read_songs_csv(path)

    def test_remote_audio_url_rejected(self, tmp_path):
        path = tmp_path / "songs.csv"
        write_csv(path, ["song_key", "song_name", "audio_local_path"], [
            {"song_key": "demo_001", "song_name": "A",
             "audio_local_path": "https://evil.example/x.mp3"},
        ])
        with pytest.raises(preprocess.CsvFormatError):
            preprocess.read_songs_csv(path)

    def test_absolute_audio_path_rejected(self, tmp_path):
        path = tmp_path / "songs.csv"
        write_csv(path, ["song_key", "song_name", "audio_local_path"], [
            {"song_key": "demo_001", "song_name": "A",
             "audio_local_path": "C:/Windows/x.mp3"},
        ])
        with pytest.raises(preprocess.CsvFormatError):
            preprocess.read_songs_csv(path)


class TestManifestCsv:
    def test_reads_valid_manifest(self, tmp_path):
        path = tmp_path / "manifest.csv"
        write_csv(path, ["song_key", "image_url", "image_path"], [
            {"song_key": "demo_001", "image_url": "",
             "image_path": "data/source_images/demo_001/01.jpg"},
            {"song_key": "demo_002", "image_url": "https://example.com/a.jpg",
             "image_path": ""},
        ])
        rows = preprocess.read_image_manifest(path)
        assert len(rows) == 2
        assert rows[0].image_path is not None
        assert rows[1].image_url is not None

    def test_both_url_and_path_rejected(self, tmp_path):
        path = tmp_path / "manifest.csv"
        write_csv(path, ["song_key", "image_url", "image_path"], [
            {"song_key": "demo_001", "image_url": "https://example.com/a.jpg",
             "image_path": "data/source_images/a.jpg"},
        ])
        with pytest.raises(preprocess.CsvFormatError):
            preprocess.read_image_manifest(path)

    def test_neither_url_nor_path_rejected(self, tmp_path):
        path = tmp_path / "manifest.csv"
        write_csv(path, ["song_key", "image_url", "image_path"], [
            {"song_key": "demo_001", "image_url": "", "image_path": ""},
        ])
        with pytest.raises(preprocess.CsvFormatError):
            preprocess.read_image_manifest(path)

    def test_non_http_url_rejected(self, tmp_path):
        path = tmp_path / "manifest.csv"
        write_csv(path, ["song_key", "image_url", "image_path"], [
            {"song_key": "demo_001", "image_url": "ftp://example.com/a.jpg",
             "image_path": ""},
        ])
        with pytest.raises(preprocess.CsvFormatError):
            preprocess.read_image_manifest(path)

    def test_path_outside_source_images_rejected(self, tmp_path):
        path = tmp_path / "manifest.csv"
        write_csv(path, ["song_key", "image_url", "image_path"], [
            {"song_key": "demo_001", "image_url": "",
             "image_path": "C:/Windows/x.jpg"},
        ])
        with pytest.raises(preprocess.CsvFormatError):
            preprocess.read_image_manifest(path)


class TestDatabaseUpsert:
    def test_song_inserted_once(self, temp_db):
        with db.get_connection() as conn:
            song_id_1, created_1 = db.upsert_song(
                conn, song_key="demo_001", song_name="A"
            )
            song_id_2, created_2 = db.upsert_song(
                conn, song_key="demo_001", song_name="A 改名"
            )
        assert created_1 is True
        assert created_2 is False
        assert song_id_1 == song_id_2  # 重复导入不改变 id

    def test_same_image_not_duplicated(self, temp_db):
        vector = [0.0] * config.EMBEDDING_DIM
        vector[0] = 1.0

        with db.get_connection() as conn:
            song_id, _ = db.upsert_song(conn, song_key="demo_001", song_name="A")
            _, created_1 = db.upsert_comment_image(
                conn, song_id=song_id, image_local_path="comment_images/a.jpg",
                source_url=None, image_sha256="abc", embedding=vector,
                embedding_model=config.CLIP_MODEL_NAME,
            )
            _, created_2 = db.upsert_comment_image(
                conn, song_id=song_id, image_local_path="comment_images/a.jpg",
                source_url=None, image_sha256="abc", embedding=vector,
                embedding_model=config.CLIP_MODEL_NAME,
            )
            rows = db.list_images_for_songs(conn, [song_id])

        assert created_1 is True
        assert created_2 is False
        assert len(rows) == 1

    def test_different_image_same_song_allowed(self, temp_db):
        vector = [0.0] * config.EMBEDDING_DIM
        vector[0] = 1.0

        with db.get_connection() as conn:
            song_id, _ = db.upsert_song(conn, song_key="demo_001", song_name="A")
            db.upsert_comment_image(
                conn, song_id=song_id, image_local_path="comment_images/a.jpg",
                source_url=None, image_sha256="aaa", embedding=vector,
                embedding_model=config.CLIP_MODEL_NAME,
            )
            db.upsert_comment_image(
                conn, song_id=song_id, image_local_path="comment_images/b.jpg",
                source_url=None, image_sha256="bbb", embedding=vector,
                embedding_model=config.CLIP_MODEL_NAME,
            )
            rows = db.list_images_for_songs(conn, [song_id])

        assert len(rows) == 2

    def test_foreign_key_enforced(self, temp_db):
        vector = [0.0] * config.EMBEDDING_DIM
        vector[0] = 1.0
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                db.upsert_comment_image(
                    conn, song_id=99999, image_local_path="comment_images/a.jpg",
                    source_url=None, image_sha256="zzz", embedding=vector,
                    embedding_model=config.CLIP_MODEL_NAME,
                )


class TestRebuildPreserves:
    def test_rebuild_keeps_song_id_and_posts(self, temp_db):
        """重建不改变歌曲主键，也不删除留言。"""
        vector = [0.0] * config.EMBEDDING_DIM
        vector[0] = 1.0

        with db.get_connection() as conn:
            song_id, _ = db.upsert_song(conn, song_key="demo_001", song_name="A")
            db.upsert_comment_image(
                conn, song_id=song_id, image_local_path="comment_images/old.jpg",
                source_url=None, image_sha256="old", embedding=vector,
                embedding_model=config.CLIP_MODEL_NAME,
            )
            db.insert_post(
                conn, song_id=song_id, upload_image_path="uploads/u.jpg",
                nickname="测试", content="留言内容",
            )

        # 模拟重建：替换图片记录
        with db.get_connection() as conn:
            old = db.list_images_for_songs(conn, [song_id])
            db.delete_comment_images_by_ids(conn, [int(r["id"]) for r in old])
            db.upsert_comment_image(
                conn, song_id=song_id, image_local_path="comment_images/new.jpg",
                source_url=None, image_sha256="new", embedding=vector,
                embedding_model=config.CLIP_MODEL_NAME,
            )

        with db.get_connection() as conn:
            song = db.get_song_by_id(conn, song_id)
            posts = db.list_posts_for_song(conn, song_id)
            images = db.list_images_for_songs(conn, [song_id])

        assert song is not None and int(song["id"]) == song_id  # 主键不变
        assert len(posts) == 1                                   # 留言保留
        assert len(images) == 1
        assert images[0]["image_sha256"] == "new"
