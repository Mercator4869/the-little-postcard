"""离线预处理程序（计划书 §10）。

职责：初始化表 → 导入歌曲 → 读取图片清单 → 读取/下载图片 →
校验规范化 → 提取 CLIP 特征 → 保存图片与向量 → 输出统计。

预处理是独立批处理程序，不得由 Flask 请求调用。

用法：
    python preprocess.py --seed data/songs_seed.csv \
        --image-manifest data/comment_images_manifest.csv
    python preprocess.py ... --rebuild
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import config
import db
import image_utils

logger = logging.getLogger("preprocess")


# ================================================================ CSV 解析
@dataclass
class SongRow:
    song_key: str
    song_name: str
    artist: str = ""
    description: str = ""
    audio_local_path: str | None = None


@dataclass
class ImageRow:
    song_key: str
    image_url: str | None = None
    image_path: str | None = None


class CsvFormatError(ValueError):
    """输入文件格式错误（返回非零退出码）。"""


def read_songs_csv(path: Path) -> list[SongRow]:
    if not path.exists():
        raise CsvFormatError(f"歌曲清单不存在：{path}")

    rows: list[SongRow] = []
    seen: set[str] = set()

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"song_key", "song_name"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise CsvFormatError(
                f"歌曲清单缺少必需列 {sorted(required)}，实际列：{reader.fieldnames}"
            )

        for line_no, raw in enumerate(reader, start=2):
            song_key = (raw.get("song_key") or "").strip()
            song_name = (raw.get("song_name") or "").strip()

            if not song_key:
                raise CsvFormatError(f"第 {line_no} 行：song_key 必填")
            if not config.is_valid_song_key(song_key):
                raise CsvFormatError(
                    f"第 {line_no} 行：song_key 只能包含字母、数字、下划线、连字符"
                )
            if song_key in seen:
                raise CsvFormatError(f"第 {line_no} 行：song_key 重复：{song_key}")
            if not song_name:
                raise CsvFormatError(f"第 {line_no} 行：song_name 必填")

            audio = (raw.get("audio_local_path") or "").strip() or None
            if audio is not None:
                if not audio.replace("\\", "/").startswith("audio/"):
                    raise CsvFormatError(
                        f"第 {line_no} 行：audio_local_path 必须位于 static/audio/ 内"
                    )
                if ".." in audio.split("/") or Path(audio).is_absolute():
                    raise CsvFormatError(f"第 {line_no} 行：audio_local_path 路径非法")

            seen.add(song_key)
            rows.append(
                SongRow(
                    song_key=song_key,
                    song_name=song_name,
                    artist=(raw.get("artist") or "").strip(),
                    description=(raw.get("description") or "").strip(),
                    audio_local_path=audio,
                )
            )

    return rows


def read_image_manifest(path: Path) -> list[ImageRow]:
    if not path.exists():
        raise CsvFormatError(f"图片清单不存在：{path}")

    rows: list[ImageRow] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "song_key" not in reader.fieldnames:
            raise CsvFormatError(f"图片清单缺少 song_key 列，实际列：{reader.fieldnames}")

        for line_no, raw in enumerate(reader, start=2):
            song_key = (raw.get("song_key") or "").strip()
            image_url = (raw.get("image_url") or "").strip()
            image_path = (raw.get("image_path") or "").strip()

            if not song_key:
                raise CsvFormatError(f"第 {line_no} 行：song_key 必填")
            if not config.is_valid_song_key(song_key):
                raise CsvFormatError(f"第 {line_no} 行：song_key 格式非法")

            has_url = bool(image_url)
            has_path = bool(image_path)
            if has_url == has_path:  # 两者都有或都没有
                raise CsvFormatError(
                    f"第 {line_no} 行：image_url 与 image_path 必须且只能填写一个"
                )

            if has_url:
                if not image_url.lower().startswith(("http://", "https://")):
                    raise CsvFormatError(f"第 {line_no} 行：URL 只允许 HTTP/HTTPS")
                rows.append(ImageRow(song_key=song_key, image_url=image_url))
            else:
                local = Path(image_path)
                if local.is_absolute() or ".." in image_path.replace("\\", "/").split("/"):
                    raise CsvFormatError(f"第 {line_no} 行：本地路径非法")
                resolved = (config.BASE_DIR / image_path).resolve()
                if not image_utils.is_within(config.SOURCE_IMAGES_DIR, resolved):
                    raise CsvFormatError(
                        f"第 {line_no} 行：本地图片必须位于 data/source_images/ 内"
                    )
                rows.append(ImageRow(song_key=song_key, image_path=image_path))

    return rows


# ================================================================ 下载
def download_image(url: str) -> bytes:
    """下载图片：超时、有限重试、大小限制；失败记录日志不无限等待。"""
    import requests

    last_error: Exception | None = None

    for attempt in range(1, config.DOWNLOAD_MAX_RETRIES + 1):
        try:
            with requests.get(
                url,
                timeout=config.DOWNLOAD_TIMEOUT_SECONDS,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0 (Image2MusicDemo/1.0)"},
            ) as response:
                response.raise_for_status()

                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > config.DOWNLOAD_MAX_BYTES:
                    raise ValueError(f"远程文件超过大小限制：{declared} 字节")

                buffer = io.BytesIO()
                total = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > config.DOWNLOAD_MAX_BYTES:
                        raise ValueError("远程文件超过大小限制")
                    buffer.write(chunk)

                return buffer.getvalue()

        except Exception as exc:
            last_error = exc
            logger.warning("下载失败（第 %d/%d 次）：%s — %s",
                           attempt, config.DOWNLOAD_MAX_RETRIES, url, exc)

    raise RuntimeError(f"下载失败：{last_error}")


# ================================================================ 结果统计
@dataclass
class Stats:
    songs_added: int = 0
    songs_updated: int = 0
    images_added: int = 0
    images_duplicate: int = 0
    images_failed: int = 0
    valid_vectors: int = 0
    failures: list[str] = field(default_factory=list)

    def report(self, elapsed: float) -> str:
        lines = [
            "=" * 56,
            "预处理统计",
            "=" * 56,
            f"歌曲新增数        : {self.songs_added}",
            f"歌曲更新数        : {self.songs_updated}",
            f"图片新增数        : {self.images_added}",
            f"图片重复跳过数    : {self.images_duplicate}",
            f"图片失败数        : {self.images_failed}",
            f"有效向量数        : {self.valid_vectors}",
            f"处理耗时          : {elapsed:.2f} 秒",
            "=" * 56,
        ]
        if self.failures:
            lines.append("失败明细：")
            lines.extend(f"  - {item}" for item in self.failures)
        return "\n".join(lines)


# ================================================================ 主流程
def run_preprocess(
    seed_path: Path,
    manifest_path: Path,
    *,
    rebuild: bool = False,
) -> tuple[int, Stats]:
    """执行预处理，返回 (退出码, 统计)。"""
    started = time.monotonic()
    stats = Stats()

    config.ensure_directories()

    # 1. 解析输入（格式错误 → 非零退出码）
    song_rows = read_songs_csv(seed_path)
    image_rows = read_image_manifest(manifest_path)

    # 2. 初始化数据库表
    db.init_db()

    # 3. 延迟加载编码器：仅在确有图片需要处理时加载
    encoder = None
    encoder_error: str | None = None

    def get_encoder():
        nonlocal encoder, encoder_error
        if encoder is not None:
            return encoder
        if encoder_error is not None:
            return None
        try:
            from clip_encoder import ClipImageEncoder

            logger.info("加载 CLIP 模型（首次较慢）…")
            encoder = ClipImageEncoder.load()
            return encoder
        except Exception as exc:
            encoder_error = str(exc)
            logger.error("模型加载失败：%s", exc)
            return None

    # 4. 导入歌曲（按 song_key 更新，不改动 id）
    song_id_by_key: dict[str, int] = {}
    with db.get_connection() as conn:
        for row in song_rows:
            song_id, created = db.upsert_song(
                conn,
                song_key=row.song_key,
                song_name=row.song_name,
                artist=row.artist,
                description=row.description,
                audio_local_path=row.audio_local_path,
            )
            song_id_by_key[row.song_key] = song_id
            if created:
                stats.songs_added += 1
            else:
                stats.songs_updated += 1

        # 清单涉及的歌曲必须存在
        for image_row in image_rows:
            if image_row.song_key not in song_id_by_key:
                existing = db.get_song_by_key(conn, image_row.song_key)
                if existing is None:
                    raise CsvFormatError(
                        f"图片清单中的歌曲不存在于歌曲清单或数据库：{image_row.song_key}"
                    )
                song_id_by_key[image_row.song_key] = int(existing["id"])

    # 5. 处理图片
    #    rebuild 模式：先写临时目录，验证通过后再替换数据库记录
    temp_root: Path | None = None
    pending: list[dict] = []          # 待写入的记录（非 rebuild）
    pending_by_song: dict[str, list[dict]] = {}

    try:
        if rebuild:
            temp_root = Path(tempfile.mkdtemp(prefix="i2m_rebuild_", dir=str(config.DATA_DIR)))
            logger.info("重建模式：临时目录 %s", temp_root)

        processed_keys: set[str] = set()

        for image_row in image_rows:
            song_key = image_row.song_key
            song_id = song_id_by_key[song_key]
            processed_keys.add(song_key)

            # 取字节
            try:
                if image_row.image_url:
                    raw = download_image(image_row.image_url)
                    source_url = image_row.image_url
                else:
                    local = (config.BASE_DIR / image_row.image_path).resolve()
                    if not local.exists():
                        raise FileNotFoundError(f"本地图片不存在：{image_row.image_path}")
                    raw = local.read_bytes()
                    source_url = None
            except Exception as exc:
                stats.images_failed += 1
                stats.failures.append(f"{song_key} / 读取失败：{exc}")
                continue

            # 校验规范化（幂等：供后续 save_comment_image 复用，避免哈希漂移）
            try:
                normalized = image_utils.normalize_once(raw)
            except image_utils.ImageValidationError as exc:
                stats.images_failed += 1
                stats.failures.append(f"{song_key} / 图片无效：{exc}")
                continue

            digest = image_utils.sha256_of_bytes(normalized)

            # 提特征
            enc = get_encoder()
            if enc is None:
                stats.images_failed += 1
                stats.failures.append(f"{song_key} / 模型不可用：{encoder_error}")
                continue

            try:
                from PIL import Image

                with Image.open(io.BytesIO(normalized)) as img:
                    vector = enc.encode_image(img.convert("RGB"))
            except Exception as exc:
                stats.images_failed += 1
                stats.failures.append(f"{song_key} / 特征提取失败：{exc}")
                continue

            record = {
                "song_key": song_key,
                "song_id": song_id,
                "digest": digest,
                "vector": vector,
                "source_url": source_url,
                "normalized": normalized,
            }

            if rebuild:
                # 写入临时目录，暂不落正式位置
                target_dir = temp_root / song_key
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / f"{digest}.jpg").write_bytes(normalized)
                record["temp_path"] = target_dir / f"{digest}.jpg"
                pending_by_song.setdefault(song_key, []).append(record)
            else:
                pending.append(record)

        stats.valid_vectors = sum(
            len(v) for v in pending_by_song.values()
        ) if rebuild else len(pending)

        # ---- 非 rebuild：直接落盘 + 落库 ----
        # 说明：文件写入与数据库写入在同一段逻辑内。若提交失败，
        # 已写出的图片会成为"无数据库引用的孤儿文件"——它们体积很小、
        # 且下次导入会因内容哈希相同而复用同一路径，因此不影响正确性；
        # 这里记录日志以便排查，不做删除（删除反而可能误删旧引用）。
        if not rebuild:
            written_paths: list[str] = []
            try:
                with db.get_connection() as conn:
                    for record in pending:
                        relative_path, digest = image_utils.save_comment_image(
                            record["normalized"], song_key=record["song_key"]
                        )
                        written_paths.append(relative_path)
                        _, created = db.upsert_comment_image(
                            conn,
                            song_id=record["song_id"],
                            image_local_path=relative_path,
                            source_url=record["source_url"],
                            image_sha256=digest,
                            embedding=record["vector"],
                            embedding_model=config.CLIP_MODEL_NAME,
                        )
                        if created:
                            stats.images_added += 1
                        else:
                            stats.images_duplicate += 1
            except Exception:
                if written_paths:
                    logger.warning(
                        "导入失败，已写出的 %d 个图片文件保留在磁盘上：%s",
                        len(written_paths),
                        "、".join(written_paths[:5]) + ("…" if len(written_paths) > 5 else ""),
                    )
                raise

        # ---- rebuild：验证后替换 ----
        else:
            _apply_rebuild(pending_by_song, processed_keys, stats)

    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)

    elapsed = time.monotonic() - started
    logger.info("\n%s", stats.report(elapsed))

    exit_code = 1 if (stats.images_failed > 0 and stats.valid_vectors == 0) else 0
    return exit_code, stats


def _apply_rebuild(
    pending_by_song: dict[str, list[dict]],
    processed_keys: set[str],
    stats: Stats,
) -> None:
    """安全重建：先验证准备结果，再替换数据库记录，最后清理旧文件。

    - 保留歌曲主键、歌曲记录、留言、用户上传图片、本地音频、其他歌曲的图片
    - 重建失败时不先清空旧索引
    - 某首歌全部处理失败时保留其旧索引并报告失败
    - 清单明确不包含图片的歌曲，允许重建为空

    文件删除**推迟到事务提交成功之后**执行：若写库失败回滚，
    旧图片文件仍然存在，不会出现"数据库有记录但文件已删"的不一致。
    """
    orphans: set[str] = set()

    with db.get_connection() as conn:
        for song_key in processed_keys:
            row = db.get_song_by_key(conn, song_key)
            if row is None:
                continue
            song_id = int(row["id"])

            prepared = pending_by_song.get(song_key, [])

            # 该歌曲在清单里有图片但全部失败 → 保留旧索引
            if not prepared:
                existing = db.list_images_for_songs(conn, [song_id])
                if existing:
                    stats.failures.append(
                        f"{song_key} / 重建时该歌曲无成功图片，保留旧索引（{len(existing)} 条）"
                    )
                continue

            # 落盘到正式位置并收集新记录
            new_records: list[dict] = []
            for record in prepared:
                relative_path, digest = image_utils.save_comment_image(
                    record["normalized"], song_key=song_key
                )
                new_records.append(
                    {
                        "relative_path": relative_path,
                        "digest": digest,
                        "vector": record["vector"],
                        "source_url": record["source_url"],
                    }
                )

            old_rows = db.list_images_for_songs(conn, [song_id])
            old_ids = [int(r["id"]) for r in old_rows]
            old_paths = {str(r["image_local_path"]) for r in old_rows}

            # 替换数据库记录
            db.delete_comment_images_by_ids(conn, old_ids)
            for record in new_records:
                db.upsert_comment_image(
                    conn,
                    song_id=song_id,
                    image_local_path=record["relative_path"],
                    source_url=record["source_url"],
                    image_sha256=record["digest"],
                    embedding=record["vector"],
                    embedding_model=config.CLIP_MODEL_NAME,
                )
                stats.images_added += 1

            # 收集不再被引用的旧文件，等事务提交成功后再删
            still_used = {r["relative_path"] for r in new_records}
            for old_path in old_paths:
                if old_path not in still_used:
                    orphans.add(old_path)

    # 事务已提交，此时才安全删除失去引用的旧文件
    for old_path in sorted(orphans):
        absolute = image_utils.static_relative_to_absolute(old_path)
        if absolute is not None and absolute.exists():
            try:
                absolute.unlink()
            except OSError as exc:
                logger.warning("清理旧图片失败：%s — %s", old_path, exc)


# ================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="图片匹配音乐 Demo — 离线预处理程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", default="data/songs_seed.csv", help="歌曲清单 CSV")
    parser.add_argument(
        "--image-manifest", default="data/comment_images_manifest.csv", help="图片清单 CSV"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="重建清单涉及的歌曲图片索引（保留歌曲主键与留言）",
    )
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 相对路径按项目目录解析，不依赖当前工作目录
    seed_path = Path(args.seed)
    if not seed_path.is_absolute():
        seed_path = config.BASE_DIR / seed_path

    manifest_path = Path(args.image_manifest)
    if not manifest_path.is_absolute():
        manifest_path = config.BASE_DIR / manifest_path

    try:
        exit_code, _ = run_preprocess(seed_path, manifest_path, rebuild=args.rebuild)
        return exit_code
    except CsvFormatError as exc:
        logger.error("输入文件格式错误：%s", exc)
        return 2
    except db.DatabaseUnavailable as exc:
        logger.error("数据库不可访问：%s", exc)
        return 3
    except Exception:
        logger.exception("预处理整体失败")
        return 4


if __name__ == "__main__":
    sys.exit(main())
