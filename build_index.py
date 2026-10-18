"""离线回填：评论配文 / 留言图向量 / 文本向量 / 情感分值。

用法（在项目目录执行）：

    python build_index.py               # 只补缺失，幂等
    python build_index.py --rebuild     # 清空派生索引后全部重算
    python build_index.py --dry-run     # 只统计，不写库
    python build_index.py --limit 5     # 每类只处理 5 条（调试用）

产物：
- comment_image_notes：评论图 → 笔记正文（来自 data/source_images/**/notes.json）
- post_image_embeddings：留言图 → CLIP 512 维向量
- item_texts：评论图 / 留言的 bge 文本向量 + StructBERT 情感 valence

运行期不联网；模型均从 data/model_cache 本地加载。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import db  # noqa: E402
import image_utils  # noqa: E402
from clip_encoder import ClipImageEncoder, ModelLoadError as ClipLoadError  # noqa: E402
from text_encoder import (  # noqa: E402
    ModelLoadError as TextLoadError,
    SentimentEncoder,
    TextEncoder,
)


def _log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------- 配文映射
def backfill_comment_notes(conn, *, dry_run: bool, limit: int | None) -> dict[str, int]:
    comment_paths = {
        str(row["image_local_path"]): int(row["id"])
        for row in conn.execute("SELECT id, image_local_path FROM song_comment_images")
    }
    stats = {"mapped": 0, "missing_file": 0, "not_in_db": 0, "no_text": 0, "limit": 0}
    seen: set[int] = set()

    for notes_file in sorted(config.SOURCE_IMAGES_DIR.rglob("notes.json")):
        song_key = notes_file.parent.name
        if not config.is_valid_song_key(song_key):
            _log(f"[跳过] 目录名不是合法 song_key：{song_key}")
            continue
        try:
            entries = json.loads(notes_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _log(f"[失败] 读取 {notes_file}：{exc}")
            continue

        for entry in entries:
            if limit is not None and stats["mapped"] >= limit:
                stats["limit"] = 1
                break
            text = (entry.get("note_text") or "").strip()
            image_name = (entry.get("image_file") or "").strip()
            if not text:
                stats["no_text"] += 1
                continue
            if not image_name:
                stats["missing_file"] += 1
                continue
            source = notes_file.parent / image_name
            if not source.is_file():
                stats["missing_file"] += 1
                continue
            digest = image_utils.sha256_of_bytes(
                image_utils.normalize_once(source.read_bytes())
            )
            relative = f"comment_images/{song_key}/{digest}.jpg"
            comment_image_id = comment_paths.get(relative)
            if comment_image_id is None:
                stats["not_in_db"] += 1
                continue
            if comment_image_id in seen:
                continue
            seen.add(comment_image_id)
            stats["mapped"] += 1
            if not dry_run:
                db.upsert_comment_note(
                    conn,
                    comment_image_id=comment_image_id,
                    note_text=text,
                    note_author=entry.get("note_author"),
                    note_time_ms=entry.get("note_time_ms"),
                    source_link=entry.get("source_link"),
                )
        if stats["limit"]:
            break

    _log(
        "配文映射：写入 {mapped}，无正文 {no_text}，源图缺失 {missing_file}，"
        "库中无对应图 {not_in_db}".format(**stats)
    )
    return stats


# ------------------------------------------------------------ 留言图向量
def backfill_post_images(
    conn,
    encoder,
    *,
    dry_run: bool,
    limit: int | None,
    skip_ids: set[int] | None = None,
) -> dict[str, int]:
    stats = {"done": 0, "missing": 0, "failed": 0, "skipped": 0}
    skip_ids = skip_ids or set()
    rows = list(
        conn.execute(
            "SELECT id, upload_image_path FROM posts ORDER BY id ASC"
            + (f" LIMIT {int(limit)}" if limit is not None else "")
        )
    )
    for position, row in enumerate(rows, 1):
        post_id = int(row["id"])
        if post_id in skip_ids:
            stats["skipped"] += 1
            continue
        relative = str(row["upload_image_path"])
        absolute = image_utils.static_relative_to_absolute(relative)
        if absolute is None or not absolute.exists():
            stats["missing"] += 1
            continue
        try:
            with Image.open(absolute) as image:
                vector = encoder.encode_image(image.convert("RGB"))
            digest = image_utils.sha256_of_file(absolute)
            if not dry_run:
                db.upsert_post_image_embedding(
                    conn,
                    post_id=post_id,
                    image_sha256=digest,
                    embedding=vector,
                    embedding_model=config.CLIP_MODEL_NAME,
                )
            stats["done"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            _log(f"[失败] 留言图 {post_id}：{exc}")
        if position % 25 == 0:
            _log(f"  留言图 {position}/{len(rows)} ...")
        if not dry_run:
            conn.commit()

    _log(
        "留言图向量：成功 {done}，跳过已有 {skipped}，文件缺失 {missing}，失败 {failed}".format(
            **stats
        )
    )
    return stats


# -------------------------------------------------- 文本向量 + 情感分值
def _text_tasks(conn, limit: int | None) -> list[tuple[str, int, str]]:
    tasks: list[tuple[str, int, str]] = []

    comment_rows = conn.execute(
        """
        SELECT cin.comment_image_id AS item_id, cin.note_text AS text
          FROM comment_image_notes AS cin
         ORDER BY cin.comment_image_id ASC
        """
        + (f" LIMIT {int(limit)}" if limit is not None else "")
    ).fetchall()
    for row in comment_rows:
        text = (row["text"] or "").strip()
        if text:
            tasks.append(("comment_image", int(row["item_id"]), text))

    post_rows = conn.execute(
        "SELECT id, content FROM posts ORDER BY id ASC"
        + (f" LIMIT {int(limit)}" if limit is not None else "")
    ).fetchall()
    for row in post_rows:
        text = (row["content"] or "").strip()
        if text:
            tasks.append(("post", int(row["id"]), text))

    return tasks


def backfill_texts(
    conn,
    text_encoder: TextEncoder,
    sentiment: SentimentEncoder,
    *,
    dry_run: bool,
    limit: int | None,
) -> dict[str, int]:
    tasks = _text_tasks(conn, limit)
    stats = {"done": 0, "failed": 0, "total": len(tasks)}
    if not tasks:
        _log("文本：没有需要处理的行")
        return stats

    batch_size = 16
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        texts = [text for _type, _id, text in batch]
        try:
            vectors = text_encoder.encode(texts)
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += len(batch)
            _log(f"[失败] 文本批编码 {start}：{exc}")
            continue

        for (item_type, item_id, text), vector in zip(batch, vectors):
            try:
                valence, label, probability = sentiment.score(text)
            except Exception as exc:  # noqa: BLE001
                valence, label, probability = None, None, None
                _log(f"[警告] 情感打分失败 {item_type}/{item_id}：{exc}")
            if not dry_run:
                db.upsert_item_text(
                    conn,
                    item_type=item_type,
                    item_id=item_id,
                    text=text,
                    embedding=vector,
                    text_model=config.TEXT_MODEL_NAME,
                    valence=valence,
                    emotion_label=label,
                    emotion_probability=probability,
                    sentiment_model=config.SENTIMENT_MODEL_NAME,
                )
            stats["done"] += 1

        if not dry_run:
            conn.commit()
        _log(f"  文本 {min(start + batch_size, len(tasks))}/{len(tasks)} ...")

    _log(
        "文本 + 情感：成功 {done}/{total}，失败 {failed}".format(**stats)
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回填评论配文、留言图向量、文本向量与情感分值")
    parser.add_argument("--rebuild", action="store_true", help="清空派生索引后全部重算")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写库")
    parser.add_argument("--limit", type=int, default=None, help="每类最多处理 N 条（调试）")
    args = parser.parse_args(argv)

    started = time.time()
    config.ensure_directories()
    db.init_db()

    try:
        _log("加载 CLIP ...")
        clip = ClipImageEncoder.load()
        _log("加载 bge 文本模型 ...")
        text_encoder = TextEncoder.load()
        _log("加载 StructBERT 情感模型 ...")
        sentiment = SentimentEncoder.load()
    except (ClipLoadError, TextLoadError) as exc:
        _log(f"[错误] 模型加载失败：{exc}")
        return 2

    with db.get_connection() as conn:
        if args.rebuild and not args.dry_run:
            _log("--rebuild：清空 post_image_embeddings / item_texts")
            db.clear_derived_indexes(conn)

        backfill_comment_notes(conn, dry_run=args.dry_run, limit=args.limit)
        existing_post_ids = set()
        if not args.rebuild:
            existing_post_ids = {
                int(row["post_id"])
                for row in conn.execute("SELECT post_id FROM post_image_embeddings")
            }
        backfill_post_images(
            conn,
            clip,
            dry_run=args.dry_run,
            limit=args.limit,
            skip_ids=existing_post_ids,
        )
        backfill_texts(
            conn,
            text_encoder,
            sentiment,
            dry_run=args.dry_run,
            limit=args.limit,
        )

        counts = {}
        for table in (
            "songs",
            "song_comment_images",
            "posts",
            "comment_image_notes",
            "post_image_embeddings",
            "item_texts",
        ):
            counts[table] = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

    _log("")
    _log("=== 回填完成 ===")
    for table, count in counts.items():
        _log(f"  {table}: {count}")
    _log(f"耗时 {time.time() - started:.1f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
