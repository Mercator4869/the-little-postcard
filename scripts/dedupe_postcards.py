"""Deduplicate existing postcards by song + content + author + date.

The same postcard historically existed twice: once as a user ``post`` and once
as a matchable ``song_comment_images`` note.  This script keeps the note (the
matching corpus) and removes the mirrored post, migrating any likes/replies
first.  Run with ``--apply`` to write; default is a dry run.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


def _norm(value: str | None) -> str:
    return " ".join((value or "").split())


def _note_date(ms: object) -> str:
    try:
        value = int(ms)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    try:
        return datetime.fromtimestamp(value / 1000).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def load_rows(conn: sqlite3.Connection) -> list[dict]:
    rows: list[dict] = []
    for row in conn.execute(
        """
        SELECT p.id, p.song_id, p.content, p.nickname, p.created_at,
               (SELECT COUNT(*) FROM post_likes pl WHERE pl.post_id = p.id) AS likes,
               (SELECT COUNT(*) FROM post_replies pr WHERE pr.post_id = p.id) AS replies
          FROM posts AS p
        """
    ):
        rows.append(
            {
                "kind": "post",
                "id": int(row["id"]),
                "song_id": int(row["song_id"]),
                "content": _norm(row["content"]),
                "author": _norm(row["nickname"]) or config.DEFAULT_NOTE_AUTHOR,
                "date": (row["created_at"] or "")[:10],
                "likes": int(row["likes"]),
                "replies": int(row["replies"]),
            }
        )
    for row in conn.execute(
        """
        SELECT ci.id, ci.song_id, cin.note_text, cin.note_author, cin.note_time_ms,
               (SELECT COUNT(*) FROM note_likes nl
                 WHERE nl.comment_image_id = ci.id) AS likes,
               (SELECT COUNT(*) FROM note_replies nr
                 WHERE nr.comment_image_id = ci.id) AS replies
          FROM song_comment_images AS ci
          JOIN comment_image_notes AS cin ON cin.comment_image_id = ci.id
        """
    ):
        rows.append(
            {
                "kind": "note",
                "id": int(row["id"]),
                "song_id": int(row["song_id"]),
                "content": _norm(row["note_text"]),
                "author": _norm(row["note_author"]) or config.DEFAULT_NOTE_AUTHOR,
                "date": _note_date(row["note_time_ms"]),
                "likes": int(row["likes"]),
                "replies": int(row["replies"]),
            }
        )
    return rows


def group_duplicates(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["song_id"], row["content"], row["author"], row["date"])].append(row)
    return {key: value for key, value in groups.items() if len(value) > 1}


def _delete_post(conn: sqlite3.Connection, post_id: int) -> None:
    conn.execute("DELETE FROM post_likes WHERE post_id = ?", (post_id,))
    conn.execute("DELETE FROM post_replies WHERE post_id = ?", (post_id,))
    conn.execute("DELETE FROM post_image_embeddings WHERE post_id = ?", (post_id,))
    conn.execute(
        "DELETE FROM item_texts WHERE item_type = 'post' AND item_id = ?",
        (post_id,),
    )
    conn.execute("DELETE FROM posts WHERE id = ?", (post_id,))


def _delete_note(conn: sqlite3.Connection, note_id: int) -> None:
    conn.execute("DELETE FROM note_likes WHERE comment_image_id = ?", (note_id,))
    conn.execute("DELETE FROM note_replies WHERE comment_image_id = ?", (note_id,))
    conn.execute(
        "DELETE FROM item_texts WHERE item_type = 'comment_image' AND item_id = ?",
        (note_id,),
    )
    conn.execute("DELETE FROM comment_image_notes WHERE comment_image_id = ?", (note_id,))
    conn.execute("DELETE FROM song_comment_images WHERE id = ?", (note_id,))


def _migrate_post_to_note(conn: sqlite3.Connection, post_id: int, note_id: int) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO note_likes (comment_image_id, visitor_key, created_at)
        SELECT ?, visitor_key, created_at FROM post_likes WHERE post_id = ?
        """,
        (note_id, post_id),
    )
    conn.execute(
        """
        INSERT INTO note_replies (comment_image_id, nickname, content, created_at)
        SELECT ?, nickname, content, created_at FROM post_replies WHERE post_id = ?
        """,
        (note_id, post_id),
    )


def _migrate_note_to_note(conn: sqlite3.Connection, old_id: int, note_id: int) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO note_likes (comment_image_id, visitor_key, created_at)
        SELECT ?, visitor_key, created_at FROM note_likes WHERE comment_image_id = ?
        """,
        (note_id, old_id),
    )
    conn.execute(
        """
        INSERT INTO note_replies (comment_image_id, nickname, content, created_at)
        SELECT ?, nickname, content, created_at FROM note_replies
         WHERE comment_image_id = ?
        """,
        (note_id, old_id),
    )


def dedupe(conn: sqlite3.Connection, *, apply: bool) -> dict:
    groups = group_duplicates(load_rows(conn))
    removed: list[dict] = []
    for _key, group in groups.items():
        notes = [item for item in group if item["kind"] == "note"]
        posts = [item for item in group if item["kind"] == "post"]
        notes.sort(key=lambda item: (item["likes"] + item["replies"], item["id"]), reverse=True)
        posts.sort(key=lambda item: (item["likes"] + item["replies"], item["id"]), reverse=True)

        if notes:
            canonical = notes[0]
            for note in notes[1:]:
                removed.append({**note, "canonical_kind": "note", "canonical_id": canonical["id"]})
                if apply:
                    _migrate_note_to_note(conn, note["id"], canonical["id"])
                    _delete_note(conn, note["id"])
            for post in posts:
                removed.append({**post, "canonical_kind": "note", "canonical_id": canonical["id"]})
                if apply:
                    _migrate_post_to_note(conn, post["id"], canonical["id"])
                    _delete_post(conn, post["id"])
        else:
            canonical = posts[0]
            for post in posts[1:]:
                removed.append({**post, "canonical_kind": "post", "canonical_id": canonical["id"]})
                if apply:
                    _delete_post(conn, post["id"])

    return {
        "groups": len(groups),
        "removed": len(removed),
        "removed_rows": removed,
        "apply": apply,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes")
    args = parser.parse_args()

    conn = sqlite3.connect(str(config.DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        result = dedupe(conn, apply=args.apply)
        if args.apply:
            conn.commit()
        else:
            conn.rollback()
    finally:
        conn.close()

    print(
        f"groups={result['groups']} extra_rows={result['removed']} "
        f"mode={'apply' if args.apply else 'dry-run'}"
    )
    for row in result["removed_rows"][:10]:
        print(
            f"  remove {row['kind']} {row['id']} -> "
            f"{row['canonical_kind']} {row['canonical_id']} "
            f"(song {row['song_id']}, {row['date']})"
        )
    if len(result["removed_rows"]) > 10:
        print(f"  ... and {len(result['removed_rows']) - 10} more")

    if args.apply:
        backup = config.DATA_DIR / f"dedupe_removed_{datetime.now():%Y%m%d_%H%M%S}.json"
        backup.write_text(
            json.dumps(result["removed_rows"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"removed-row log: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
