"""SQLite 访问层。

规范（计划书 §7.5）：
- 每个连接 PRAGMA foreign_keys = ON
- 参数化 SQL
- 批量导入使用事务
- 数据库异常记录日志，不把 SQL / 堆栈展示给用户
- 不通过删除整个数据库实现重建素材
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import config

logger = logging.getLogger(__name__)

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        song_key TEXT NOT NULL UNIQUE,
        song_name TEXT NOT NULL,
        artist TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT '',
        audio_local_path TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS song_comment_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        song_id INTEGER NOT NULL,
        image_local_path TEXT NOT NULL,
        source_url TEXT,
        image_sha256 TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        embedding_model TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (song_id) REFERENCES songs(id),
        UNIQUE(song_id, image_sha256)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        song_id INTEGER NOT NULL,
        upload_image_path TEXT NOT NULL,
        nickname TEXT NOT NULL DEFAULT '风外客',
        content TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (song_id) REFERENCES songs(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_comment_images_song_id ON song_comment_images(song_id)",
    "CREATE INDEX IF NOT EXISTS idx_posts_song_id ON posts(song_id)",
    "CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at)",
    """
    CREATE TABLE IF NOT EXISTS comment_image_notes (
        comment_image_id INTEGER PRIMARY KEY,
        note_text TEXT NOT NULL,
        note_author TEXT,
        note_time_ms INTEGER,
        source_link TEXT,
        FOREIGN KEY (comment_image_id) REFERENCES song_comment_images(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS post_image_embeddings (
        post_id INTEGER PRIMARY KEY,
        image_sha256 TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        embedding_model TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS item_texts (
        item_type TEXT NOT NULL,
        item_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        text_embedding_json TEXT,
        text_model TEXT,
        valence REAL,
        emotion_label TEXT,
        emotion_probability REAL,
        emotion_model TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (item_type, item_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS note_likes (
        comment_image_id INTEGER NOT NULL,
        visitor_key TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (comment_image_id, visitor_key),
        FOREIGN KEY (comment_image_id) REFERENCES song_comment_images(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS note_replies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        comment_image_id INTEGER NOT NULL,
        nickname TEXT NOT NULL DEFAULT '风外客',
        content TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (comment_image_id) REFERENCES song_comment_images(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_note_replies_image ON note_replies(comment_image_id)",
    """
    CREATE TABLE IF NOT EXISTS post_likes (
        post_id INTEGER NOT NULL,
        visitor_key TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (post_id, visitor_key),
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS post_replies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER NOT NULL,
        nickname TEXT NOT NULL DEFAULT '风外客',
        content TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_post_replies_post ON post_replies(post_id)",
)


class DatabaseUnavailable(RuntimeError):
    """数据库不可访问（用于返回友好 503 页面）。"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else config.DB_PATH
    # mkdir 可能抛 OSError（权限不足 / 路径被占用 / 父级不是目录），
    # 这类错误同样属于"数据库不可用"，必须能被 get_connection 归一化。
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """上下文管理器：正常提交，任何异常都回滚，连接失败转为 DatabaseUnavailable。

    回滚覆盖**所有**异常类型，而不只是 sqlite3.Error——否则非 SQL 异常
    （例如图片写盘失败）逃逸时事务处于未定义状态，只能依赖关闭连接
    来隐式回滚，那是不可依赖的行为。

    连接阶段的失败同时覆盖 sqlite3.Error 与 OSError：
    建目录失败、权限不足、磁盘只读等都无法打开数据库，
    上层应统一按 503 处理，而不是抛出未捕获的 OSError。
    """
    try:
        conn = _connect(db_path)
    except (sqlite3.Error, OSError) as exc:
        logger.error("数据库连接失败: %s", exc)
        raise DatabaseUnavailable("数据库连接失败") from exc

    try:
        yield conn
    except BaseException:
        try:
            conn.rollback()
        except sqlite3.Error:
            logger.exception("回滚失败")
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    """创建表与索引（幂等）。"""
    with get_connection(db_path) as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)


# ------------------------------------------------------------------ 歌曲
def upsert_song(
    conn: sqlite3.Connection,
    *,
    song_key: str,
    song_name: str,
    artist: str = "",
    description: str = "",
    audio_local_path: str | None = None,
) -> tuple[int, bool]:
    """按 song_key 新增或更新歌曲，返回 (song_id, created)。

    重复导入按 song_key 更新，不改变既有 id。
    """
    row = conn.execute(
        "SELECT id FROM songs WHERE song_key = ?", (song_key,)
    ).fetchone()

    if row is None:
        cursor = conn.execute(
            """
            INSERT INTO songs (song_key, song_name, artist, description, audio_local_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (song_key, song_name, artist, description, audio_local_path),
        )
        return int(cursor.lastrowid), True

    song_id = int(row["id"])
    conn.execute(
        """
        UPDATE songs
           SET song_name = ?,
               artist = ?,
               description = ?,
               audio_local_path = ?,
               updated_at = CURRENT_TIMESTAMP
         WHERE id = ?
        """,
        (song_name, artist, description, audio_local_path, song_id),
    )
    return song_id, False


def get_song_by_key(conn: sqlite3.Connection, song_key: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM songs WHERE song_key = ?", (song_key,)
    ).fetchone()


def get_song_by_id(conn: sqlite3.Connection, song_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM songs WHERE id = ?", (song_id,)).fetchone()


def song_exists(conn: sqlite3.Connection, song_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM songs WHERE id = ?", (song_id,)
    ).fetchone() is not None


# ------------------------------------------------------------ 评论图片向量
def upsert_comment_image(
    conn: sqlite3.Connection,
    *,
    song_id: int,
    image_local_path: str,
    source_url: str | None,
    image_sha256: str,
    embedding: Sequence[float],
    embedding_model: str,
) -> tuple[int, bool]:
    """同一歌曲下相同内容图片不重复插入，返回 (image_id, created)。

    若已存在且模型版本一致，则复用既有记录。
    """
    existing = conn.execute(
        """
        SELECT id, embedding_model FROM song_comment_images
         WHERE song_id = ? AND image_sha256 = ?
        """,
        (song_id, image_sha256),
    ).fetchone()

    embedding_json = json.dumps([float(x) for x in embedding])

    if existing is not None:
        image_id = int(existing["id"])
        if existing["embedding_model"] != embedding_model:
            conn.execute(
                """
                UPDATE song_comment_images
                   SET image_local_path = ?, source_url = ?,
                       embedding_json = ?, embedding_model = ?
                 WHERE id = ?
                """,
                (image_local_path, source_url, embedding_json, embedding_model, image_id),
            )
        return image_id, False

    cursor = conn.execute(
        """
        INSERT INTO song_comment_images
            (song_id, image_local_path, source_url, image_sha256,
             embedding_json, embedding_model)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (song_id, image_local_path, source_url, image_sha256, embedding_json, embedding_model),
    )
    return int(cursor.lastrowid), True


def load_comment_image_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """按评论图片 ID 升序加载（计划书 §11.4 要求顺序稳定）。"""
    return list(
        conn.execute(
            """
            SELECT id, song_id, image_local_path, embedding_json, embedding_model
              FROM song_comment_images
             ORDER BY id ASC
            """
        ).fetchall()
    )


def delete_comment_images_by_ids(
    conn: sqlite3.Connection, image_ids: Iterable[int]
) -> None:
    ids = [int(i) for i in image_ids]
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"DELETE FROM song_comment_images WHERE id IN ({placeholders})", ids
    )


def list_images_for_songs(
    conn: sqlite3.Connection, song_ids: Sequence[int]
) -> list[sqlite3.Row]:
    ids = [int(i) for i in song_ids]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    return list(
        conn.execute(
            f"""
            SELECT id, song_id, image_local_path, image_sha256
              FROM song_comment_images
             WHERE song_id IN ({placeholders})
             ORDER BY id ASC
            """,
            ids,
        ).fetchall()
    )


# ------------------------------------------------------------------ 留言
def insert_post(
    conn: sqlite3.Connection,
    *,
    song_id: int,
    upload_image_path: str,
    nickname: str,
    content: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO posts (song_id, upload_image_path, nickname, content)
        VALUES (?, ?, ?, ?)
        """,
        (song_id, upload_image_path, nickname, content),
    )
    return int(cursor.lastrowid)


def count_posts(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM posts").fetchone()
    return int(row["n"]) if row else 0


def delete_post(conn: sqlite3.Connection, post_id: int) -> bool:
    """删除一条留言（开发者模式用）。返回是否真的删掉了一条。"""
    cursor = conn.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    return cursor.rowcount > 0


def get_post(conn: sqlite3.Connection, post_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, song_id, upload_image_path, nickname, content, created_at "
        "  FROM posts WHERE id = ?",
        (post_id,),
    ).fetchone()


def list_posts_paginated(
    conn: sqlite3.Connection, *, page: int, page_size: int
) -> list[sqlite3.Row]:
    """按发布时间倒序分页，联结歌曲信息。"""
    offset = max(page - 1, 0) * page_size
    return list(
        conn.execute(
            """
            SELECT p.id, p.song_id, p.upload_image_path, p.nickname,
                   p.content, p.created_at,
                   s.song_name, s.artist, s.song_key
              FROM posts AS p
              JOIN songs AS s ON s.id = p.song_id
             ORDER BY datetime(p.created_at) DESC, p.id DESC
             LIMIT ? OFFSET ?
            """,
            (page_size, offset),
        ).fetchall()
    )


def _wall_like_pattern(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def count_wall_cards(conn: sqlite3.Connection, *, query: str | None = None) -> int:
    """星空明信片总数：用户留言 + 素材笔记；可按歌名/歌手过滤。"""
    pattern = _wall_like_pattern(query) if query else None
    post_filter_sql = ""
    note_filter_sql = "WHERE (cin.source_link IS NULL OR cin.source_link NOT LIKE 'post:%')"
    params: list = []
    if pattern:
        song_filter = "WHERE (s.song_name LIKE ? ESCAPE '\\' OR s.artist LIKE ? ESCAPE '\\')"
        post_filter_sql = song_filter
        note_filter_sql = (
            song_filter
            + " AND (cin.source_link IS NULL OR cin.source_link NOT LIKE 'post:%')"
        )
        params = [pattern, pattern]
    row = conn.execute(
        f"""
        SELECT
            (SELECT COUNT(*)
               FROM posts AS p
               JOIN songs AS s ON s.id = p.song_id
               {post_filter_sql})
          + (SELECT COUNT(*)
               FROM song_comment_images AS ci
               JOIN comment_image_notes AS cin ON cin.comment_image_id = ci.id
               JOIN songs AS s ON s.id = ci.song_id
               {note_filter_sql}) AS n
        """,
        params + params,
    ).fetchone()
    return int(row["n"]) if row else 0


def list_wall_cards(
    conn: sqlite3.Connection,
    *,
    page: int,
    page_size: int,
    query: str | None = None,
) -> list[sqlite3.Row]:
    """星空墙：用户留言与素材笔记按时间倒序混排；可按歌名/歌手过滤。"""
    offset = max(page - 1, 0) * page_size
    pattern = _wall_like_pattern(query) if query else None
    post_filter_sql = ""
    note_filter_sql = "WHERE (cin.source_link IS NULL OR cin.source_link NOT LIKE 'post:%')"
    params: list = []
    if pattern:
        song_filter = "WHERE (s.song_name LIKE ? ESCAPE '\\' OR s.artist LIKE ? ESCAPE '\\')"
        post_filter_sql = song_filter
        note_filter_sql = (
            song_filter
            + " AND (cin.source_link IS NULL OR cin.source_link NOT LIKE 'post:%')"
        )
        params = [pattern, pattern]
    sql = f"""
        SELECT * FROM (
            SELECT 'post' AS kind,
                   p.id AS item_id,
                   p.upload_image_path AS image_path,
                   p.nickname AS author,
                   p.content AS content,
                   datetime(p.created_at) AS sort_time,
                   s.id AS song_id,
                   s.song_name,
                   s.artist,
                   p.id AS post_id
              FROM posts AS p
              JOIN songs AS s ON s.id = p.song_id
              {post_filter_sql}
            UNION ALL
            SELECT 'note',
                   ci.id,
                   ci.image_local_path,
                   COALESCE(NULLIF(TRIM(cin.note_author), ''), '风外客'),
                   cin.note_text,
                   COALESCE(datetime(cin.note_time_ms / 1000, 'unixepoch'),
                            datetime(ci.created_at)),
                   s.id,
                   s.song_name,
                   s.artist,
                   NULL
              FROM song_comment_images AS ci
              JOIN comment_image_notes AS cin ON cin.comment_image_id = ci.id
              JOIN songs AS s ON s.id = ci.song_id
              {note_filter_sql}
        )
        ORDER BY sort_time DESC, item_id DESC
        LIMIT ? OFFSET ?
    """
    return list(
        conn.execute(sql, params + params + [page_size, offset]).fetchall()
    )


def list_posts_for_song(conn: sqlite3.Connection, song_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT id, song_id, upload_image_path, nickname, content, created_at
              FROM posts
             WHERE song_id = ?
             ORDER BY datetime(created_at) DESC, id DESC
            """,
            (song_id,),
        ).fetchall()
    )


# ------------------------------------------- 混合检索：配文 / 图文 / 情绪
def upsert_comment_note(
    conn: sqlite3.Connection,
    *,
    comment_image_id: int,
    note_text: str,
    note_author: str | None = None,
    note_time_ms: int | None = None,
    source_link: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO comment_image_notes
            (comment_image_id, note_text, note_author, note_time_ms, source_link)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(comment_image_id) DO UPDATE SET
            note_text = excluded.note_text,
            note_author = excluded.note_author,
            note_time_ms = excluded.note_time_ms,
            source_link = excluded.source_link
        """,
        (comment_image_id, note_text, note_author, note_time_ms, source_link),
    )


def upsert_post_image_embedding(
    conn: sqlite3.Connection,
    *,
    post_id: int,
    image_sha256: str,
    embedding: Sequence[float],
    embedding_model: str,
) -> None:
    conn.execute(
        """
        INSERT INTO post_image_embeddings
            (post_id, image_sha256, embedding_json, embedding_model)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(post_id) DO UPDATE SET
            image_sha256 = excluded.image_sha256,
            embedding_json = excluded.embedding_json,
            embedding_model = excluded.embedding_model
        """,
        (post_id, image_sha256, json.dumps([float(x) for x in embedding]), embedding_model),
    )


def upsert_item_text(
    conn: sqlite3.Connection,
    *,
    item_type: str,
    item_id: int,
    text: str,
    embedding: Sequence[float] | None,
    text_model: str | None,
    valence: float | None,
    emotion_label: str | None,
    emotion_probability: float | None,
    sentiment_model: str | None,
) -> None:
    embedding_json = (
        json.dumps([float(x) for x in embedding]) if embedding is not None else None
    )
    conn.execute(
        """
        INSERT INTO item_texts
            (item_type, item_id, text, text_embedding_json, text_model,
             valence, emotion_label, emotion_probability, emotion_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(item_type, item_id) DO UPDATE SET
            text = excluded.text,
            text_embedding_json = excluded.text_embedding_json,
            text_model = excluded.text_model,
            valence = excluded.valence,
            emotion_label = excluded.emotion_label,
            emotion_probability = excluded.emotion_probability,
            emotion_model = excluded.emotion_model
        """,
        (
            item_type,
            item_id,
            text,
            embedding_json,
            text_model,
            valence,
            emotion_label,
            emotion_probability,
            sentiment_model,
        ),
    )


def load_comment_index_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """歌曲池：评论图 + 配文（笔记正文）+ 文本向量/情绪。"""
    return list(
        conn.execute(
            """
            SELECT ci.id AS item_id,
                   ci.song_id,
                   ci.image_local_path,
                   ci.embedding_json,
                   ci.embedding_model,
                   cin.note_text AS note_text,
                   it.text_embedding_json,
                   it.text_model,
                   it.valence,
                   it.emotion_label,
                   it.emotion_probability,
                   it.emotion_model
              FROM song_comment_images AS ci
              LEFT JOIN comment_image_notes AS cin
                     ON cin.comment_image_id = ci.id
              LEFT JOIN item_texts AS it
                     ON it.item_type = 'comment_image' AND it.item_id = ci.id
             ORDER BY ci.id ASC
            """
        ).fetchall()
    )


def load_post_index_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """留言池：留言图 + 正文 + 文本向量/情绪。"""
    return list(
        conn.execute(
            """
            SELECT p.id AS item_id,
                   p.song_id,
                   p.upload_image_path AS image_local_path,
                   p.content AS note_text,
                   pie.embedding_json,
                   pie.embedding_model,
                   it.text_embedding_json,
                   it.text_model,
                   it.valence,
                   it.emotion_label,
                   it.emotion_probability,
                   it.emotion_model
              FROM posts AS p
              LEFT JOIN post_image_embeddings AS pie
                     ON pie.post_id = p.id
              LEFT JOIN item_texts AS it
                     ON it.item_type = 'post' AND it.item_id = p.id
             ORDER BY p.id ASC
            """
        ).fetchall()
    )


def get_post_with_song(conn: sqlite3.Connection, post_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT p.id, p.song_id, p.upload_image_path, p.nickname, p.content,
               p.created_at,
               s.song_name, s.artist, s.song_key, s.description, s.audio_local_path
          FROM posts AS p
          JOIN songs AS s ON s.id = p.song_id
         WHERE p.id = ?
        """,
        (post_id,),
    ).fetchone()


def get_comment_note_text(
    conn: sqlite3.Connection, comment_image_id: int
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT note_text, note_author, note_time_ms, source_link
          FROM comment_image_notes
         WHERE comment_image_id = ?
        """,
        (comment_image_id,),
    ).fetchone()


def get_note_card_with_song(conn: sqlite3.Connection, comment_image_id: int):
    """星空里来自素材笔记的一张明信片（图片 + 笔记 + 歌曲）。"""
    return conn.execute(
        """
        SELECT ci.id AS comment_image_id,
               ci.song_id,
               ci.image_local_path,
               ci.created_at,
               cin.note_text,
               cin.note_author,
               cin.note_time_ms,
               s.id AS song_id,
               s.song_name,
               s.artist,
               s.song_key,
               s.description,
               s.audio_local_path
          FROM song_comment_images AS ci
          JOIN comment_image_notes AS cin
            ON cin.comment_image_id = ci.id
          JOIN songs AS s ON s.id = ci.song_id
         WHERE ci.id = ?
        """,
        (comment_image_id,),
    ).fetchone()


def list_song_note_images(conn: sqlite3.Connection, song_id: int) -> list[sqlite3.Row]:
    """歌曲空间：该歌曲下所有「图片 + 笔记」组合（只取有配文的）。"""
    return list(
        conn.execute(
            """
            SELECT ci.id AS comment_image_id,
                   ci.image_local_path,
                   cin.note_text,
                   cin.note_author,
                   cin.note_time_ms
              FROM song_comment_images AS ci
              JOIN comment_image_notes AS cin
                ON cin.comment_image_id = ci.id
             WHERE ci.song_id = ?
             ORDER BY ci.id ASC
            """,
            (song_id,),
        ).fetchall()
    )


# ------------------------------------------------ 歌曲空间：共鸣与回复
def note_song_id(conn: sqlite3.Connection, comment_image_id: int) -> int | None:
    row = conn.execute(
        "SELECT song_id FROM song_comment_images WHERE id = ?", (comment_image_id,)
    ).fetchone()
    return int(row["song_id"]) if row is not None else None


def toggle_note_like(
    conn: sqlite3.Connection, *, comment_image_id: int, visitor_key: str
) -> bool:
    """同一个访客再点一次即取消；返回操作后是否处于「已共鸣」。"""
    existing = conn.execute(
        """
        SELECT 1 FROM note_likes
         WHERE comment_image_id = ? AND visitor_key = ?
        """,
        (comment_image_id, visitor_key),
    ).fetchone()
    if existing is not None:
        conn.execute(
            "DELETE FROM note_likes WHERE comment_image_id = ? AND visitor_key = ?",
            (comment_image_id, visitor_key),
        )
        return False
    conn.execute(
        """
        INSERT INTO note_likes (comment_image_id, visitor_key)
        VALUES (?, ?)
        """,
        (comment_image_id, visitor_key),
    )
    return True


def note_like_count(conn: sqlite3.Connection, comment_image_id: int) -> int:
    """当前这张明信片收到的共鸣数量。"""
    row = conn.execute(
        "SELECT COUNT(*) AS total FROM note_likes WHERE comment_image_id = ?",
        (comment_image_id,),
    ).fetchone()
    return int(row["total"]) if row is not None else 0


def note_like_counts(conn: sqlite3.Connection, song_id: int) -> dict[int, int]:
    rows = conn.execute(
        """
        SELECT nl.comment_image_id AS image_id, COUNT(*) AS total
          FROM note_likes AS nl
          JOIN song_comment_images AS ci ON ci.id = nl.comment_image_id
         WHERE ci.song_id = ?
         GROUP BY nl.comment_image_id
        """,
        (song_id,),
    ).fetchall()
    return {int(row["image_id"]): int(row["total"]) for row in rows}


def liked_note_ids(
    conn: sqlite3.Connection, visitor_key: str, image_ids: list[int]
) -> set[int]:
    if not image_ids:
        return set()
    placeholders = ",".join("?" for _ in image_ids)
    rows = conn.execute(
        f"SELECT comment_image_id FROM note_likes "
        f"WHERE visitor_key = ? AND comment_image_id IN ({placeholders})",
        (visitor_key, *image_ids),
    ).fetchall()
    return {int(row[0]) for row in rows}


def add_note_reply(
    conn: sqlite3.Connection,
    *,
    comment_image_id: int,
    nickname: str,
    content: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO note_replies (comment_image_id, nickname, content)
        VALUES (?, ?, ?)
        """,
        (comment_image_id, nickname, content),
    )
    return int(cursor.lastrowid)


def note_replies_for_song(conn: sqlite3.Connection, song_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT nr.comment_image_id, nr.nickname, nr.content, nr.created_at
              FROM note_replies AS nr
              JOIN song_comment_images AS ci ON ci.id = nr.comment_image_id
             WHERE ci.song_id = ?
             ORDER BY nr.id ASC
            """,
            (song_id,),
        ).fetchall()
    )


def post_like_counts(conn: sqlite3.Connection, song_id: int) -> dict[int, int]:
    rows = conn.execute(
        """
        SELECT pl.post_id, COUNT(*) AS n
          FROM post_likes AS pl
          JOIN posts AS p ON p.id = pl.post_id
         WHERE p.song_id = ?
         GROUP BY pl.post_id
        """,
        (song_id,),
    ).fetchall()
    return {int(row["post_id"]): int(row["n"]) for row in rows}


def post_like_count(conn: sqlite3.Connection, post_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM post_likes WHERE post_id = ?", (post_id,)
    ).fetchone()
    return int(row["n"]) if row else 0


def liked_post_ids(
    conn: sqlite3.Connection, visitor_key: str, post_ids: list[int]
) -> set[int]:
    if not post_ids:
        return set()
    marks = ",".join("?" for _ in post_ids)
    rows = conn.execute(
        f"""
        SELECT post_id FROM post_likes
         WHERE visitor_key = ? AND post_id IN ({marks})
        """,
        (visitor_key, *post_ids),
    ).fetchall()
    return {int(row["post_id"]) for row in rows}


def toggle_post_like(
    conn: sqlite3.Connection, *, post_id: int, visitor_key: str
) -> bool:
    existing = conn.execute(
        "SELECT 1 FROM post_likes WHERE post_id = ? AND visitor_key = ?",
        (post_id, visitor_key),
    ).fetchone()
    if existing is not None:
        conn.execute(
            "DELETE FROM post_likes WHERE post_id = ? AND visitor_key = ?",
            (post_id, visitor_key),
        )
        return False
    conn.execute(
        "INSERT INTO post_likes (post_id, visitor_key) VALUES (?, ?)",
        (post_id, visitor_key),
    )
    return True


def add_post_reply(
    conn: sqlite3.Connection,
    *,
    post_id: int,
    nickname: str,
    content: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO post_replies (post_id, nickname, content)
        VALUES (?, ?, ?)
        """,
        (post_id, nickname, content),
    )
    return int(cursor.lastrowid)


def post_replies_for_song(conn: sqlite3.Connection, song_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT pr.post_id, pr.nickname, pr.content, pr.created_at
              FROM post_replies AS pr
              JOIN posts AS p ON p.id = pr.post_id
             WHERE p.song_id = ?
             ORDER BY pr.id ASC
            """,
            (song_id,),
        ).fetchall()
    )


def clear_derived_indexes(conn: sqlite3.Connection) -> None:
    """清空可重建的派生索引（评论图不会被删除）。"""
    conn.execute("DELETE FROM post_image_embeddings")
    conn.execute("DELETE FROM item_texts")


# --------------------------------------------------------------- 健康检查
def check_database() -> bool:
    """能否正常打开并查询数据库。"""
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        return True
    except (sqlite3.Error, DatabaseUnavailable, OSError) as exc:
        logger.error("数据库健康检查失败: %s", exc)
        return False


def table_counts() -> dict[str, int]:
    """调试用：返回各表行数。"""
    result: dict[str, int] = {}
    try:
        with get_connection() as conn:
            for table in ("songs", "song_comment_images", "posts"):
                row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
                result[table] = int(row["n"]) if row else 0
    except (sqlite3.Error, DatabaseUnavailable, OSError) as exc:
        logger.error("统计表行数失败: %s", exc)
    return result
