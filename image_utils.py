"""图片校验与规范化。

用户上传图片与素材图片复用同一套基础处理函数（计划书 §9）：
1. 检查文件大小
2. Pillow 校验内容
3. 校验真实格式为 JPEG / PNG
4. 检查像素总量
5. 处理 EXIF 方向
6. 有透明通道合成到白底
7. 转 RGB
8. 等比例缩小，最长边不超过 MAX_IMAGE_SIDE
9. 保存为标准 JPEG
10. 移除原始元数据

错误分层：ImageValidationError 表示"图片本身无效"，
与模型推理错误严格区分——前者不触发兜底歌曲（§2.4 / §14）。
"""
from __future__ import annotations

import hashlib
import io
import logging
import uuid
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError

import config

logger = logging.getLogger(__name__)

# 解压炸弹防护：像素总量由我们显式检查
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Pillow 10 起推荐 Image.Resampling.LANCZOS；旧版本只有 Image.LANCZOS。
# 这里做兼容，避免因 Pillow 版本差异导致 AttributeError。
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    _LANCZOS = Image.LANCZOS


class ImageValidationError(ValueError):
    """图片无效（格式、损坏、超限等）。"""


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_normalize(data: bytes, *, max_bytes: int | None = None) -> bytes:
    """校验并规范化图片字节流，返回标准 JPEG 字节。

    抛出 ImageValidationError 表示图片无效。
    """
    limit = config.MAX_UPLOAD_FILE_BYTES if max_bytes is None else max_bytes

    if not data:
        raise ImageValidationError("文件内容为空")

    if len(data) > limit:
        limit_mib = limit / (1024 * 1024)
        raise ImageValidationError(f"文件大小超过限制（最大 {limit_mib:.0f} MiB）")

    # 不能只根据扩展名判断，必须真实解码
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # 检测损坏 / 截断
    except UnidentifiedImageError as exc:
        raise ImageValidationError("无法识别的图片格式") from exc
    except Exception as exc:  # Pillow 抛出的各类解码错误
        raise ImageValidationError("图片已损坏或无法解码") from exc

    # verify() 之后必须重新打开
    try:
        with Image.open(io.BytesIO(data)) as img:
            actual_format = (img.format or "").upper()
            if actual_format not in config.ALLOWED_UPLOAD_FORMATS:
                raise ImageValidationError(
                    f"不支持的图片格式：{actual_format or '未知'}（仅支持 JPEG、PNG）"
                )

            width, height = img.size
            if width <= 0 or height <= 0:
                raise ImageValidationError("图片尺寸无效")

            if width * height > config.MAX_IMAGE_PIXELS:
                raise ImageValidationError(
                    f"图片像素总量超过限制（最大 {config.MAX_IMAGE_PIXELS:,} 像素）"
                )

            # EXIF 方向
            img = _apply_exif_orientation(img)

            # 透明通道合成到白底
            img = _flatten_to_white(img)

            img = img.convert("RGB")

            # 等比例缩小
            img = _resize_within(img, config.MAX_IMAGE_SIDE)

            # 保存为标准 JPEG，不带原始元数据
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=90, optimize=True)
            return output.getvalue()

    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError("图片处理失败") from exc


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(img) or img
    except Exception:
        return img


def _flatten_to_white(img: Image.Image) -> Image.Image:
    """有透明通道时合成到白色背景。"""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        merged = Image.alpha_composite(background, rgba)
        return merged.convert("RGB")
    return img


def _resize_within(img: Image.Image, max_side: int) -> Image.Image:
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return img.resize(new_size, _LANCZOS)


# ------------------------------------------------------------------ 落盘
def save_upload_image(data: bytes) -> str:
    """保存用户上传图片，返回相对 static/ 的路径（如 uploads/xxx.jpg）。

    用户文件名不参与最终存储路径，使用随机 UUID。
    """
    normalized = normalize_once(data)
    config.ensure_directories()

    filename = f"{uuid.uuid4().hex}.jpg"
    target = config.UPLOAD_DIR / filename
    target.write_bytes(normalized)

    # 数据库统一使用 "/" 分隔符，且相对 static/
    return f"uploads/{filename}"


def normalize_once(data: bytes, *, max_bytes: int | None = None) -> bytes:
    """规范化图片，且对已规范化的同内容输入保持幂等。

    背景：JPEG 重编码不是幂等的。同一张图片反复走
    validate_and_normalize 会因重新量化而产生不同字节，
    哈希随之漂移，导致"数据库记录的 sha256"与"磁盘文件实际哈希"
    不一致，去重与校验都会失真。

    这里不靠前缀标记（会破坏 JPEG 文件头），而是检查输入是否已经
    满足全部规范化后置条件（JPEG、RGB、尺寸合规、无 EXIF）：
    满足则原样复用，不再重编码。

    注意：无论走哪条分支，**大小与像素上限都会被强制执行**，
    否则"已规范化"的捷径会成为绕过体积限制的漏洞。
    """
    limit = config.MAX_UPLOAD_FILE_BYTES if max_bytes is None else max_bytes

    if not data:
        raise ImageValidationError("文件内容为空")
    if len(data) > limit:
        limit_mib = limit / (1024 * 1024)
        raise ImageValidationError(f"文件大小超过限制（最大 {limit_mib:.0f} MiB）")

    if _is_already_normalized(data):
        return data
    return validate_and_normalize(data, max_bytes=limit)


def _is_already_normalized(data: bytes) -> bool:
    """判断字节是否已经是本模块规范化后的形态。

    这里必须真正**解码全部像素**，才能发现截断或损坏的 JPEG。
    注意 Pillow 的 verify() **不够**：它只校验文件结构，不解码扫描数据，
    因此对"文件头完好、尾部被截断"的 JPEG 会照常通过。
    实测：截断到 60% 的 JPEG 能通过 verify()，却在 load() 时抛出
    `OSError: image file is truncated`。
    若漏掉这一步，损坏图片会被判为"已规范化"而跳过
    validate_and_normalize 的完整性校验，用户拿到的就不是
    "图片已损坏"而是莫名其妙的兜底歌曲。
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if (probe.format or "").upper() != "JPEG":
                return False
            probe.verify()          # 结构层面：能发现部分损坏
    except Exception:
        return False

    # verify() 之后原对象不可再用，必须重新打开；
    # 并强制 load() 触发完整解码，截断/损坏在此时才会暴露。
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.mode != "RGB":
                return False
            width, height = img.size
            if width <= 0 or height <= 0:
                return False
            if width * height > config.MAX_IMAGE_PIXELS:
                return False
            if max(width, height) > config.MAX_IMAGE_SIDE:
                return False
            # 规范化后不应残留 EXIF
            if img.getexif():
                return False
            img.load()              # 截断 / 损坏在此暴露（verify() 不会）
            return True
    except Exception:
        return False


def save_comment_image(data: bytes, *, song_key: str) -> tuple[str, str]:
    """保存评论素材图片，返回 (相对路径, sha256)。

    文件名由**内容哈希**生成；路径形如 comment_images/<song_key>/<hash>.jpg
    哈希针对写入磁盘的最终字节计算，因此与文件实际内容始终一致；
    对同一输入重复调用得到相同路径与哈希。
    """
    # song_key 会直接参与文件系统路径，必须在此处自行校验，
    # 而不是依赖调用方先校验过——否则任何一次直接调用
    # （新增路由、脚本、测试）都可能写出目录之外。
    if not config.is_valid_song_key(song_key):
        raise ImageValidationError("song_key 格式非法")

    normalized = normalize_once(data)
    digest = sha256_of_bytes(normalized)

    config.ensure_directories()
    target_dir = config.COMMENT_IMAGE_DIR / song_key
    target_dir.mkdir(parents=True, exist_ok=True)

    target = target_dir / f"{digest}.jpg"
    if not target.exists():
        target.write_bytes(normalized)

    # 二次确认最终路径仍在预期目录内（纵深防御）
    if not is_within(config.COMMENT_IMAGE_DIR, target.resolve()):
        raise ImageValidationError("评论图片路径越界")

    return f"comment_images/{song_key}/{digest}.jpg", digest


# ------------------------------------------------------------- 路径安全
def static_relative_to_absolute(relative_path: str) -> Path | None:
    """把数据库中的静态相对路径解析为绝对路径。

    仅接受 STATIC_PATH_PREFIXES 开头的相对路径，并拒绝：
    绝对路径、前导斜杠、盘符、父目录 ".."、当前目录 "."、空路径段、非字符串。
    解析结果必须仍位于 STATIC_DIR 内，否则返回 None。
    """
    if not isinstance(relative_path, str) or not relative_path:
        return None

    # 统一分隔符后按原样校验，不做任何"自动修正"
    cleaned = relative_path.replace("\\", "/")

    # 拒绝绝对路径与前导斜杠（必须写成 uploads/x.jpg 这样的相对形式）
    if cleaned.startswith("/"):
        return None
    if Path(cleaned).is_absolute() or ":" in cleaned:
        return None

    # 拒绝空段与父目录穿越
    segments = cleaned.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        return None

    if not cleaned.startswith(config.STATIC_PATH_PREFIXES):
        return None

    candidate = (config.STATIC_DIR / cleaned).resolve()
    try:
        candidate.relative_to(config.STATIC_DIR.resolve())
    except ValueError:
        return None
    return candidate


def is_within(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def upload_path_is_safe(relative_path: str) -> bool:
    """留言图片路径必须位于 static/uploads/ 内。"""
    absolute = static_relative_to_absolute(relative_path)
    if absolute is None:
        return False
    if not relative_path.replace("\\", "/").startswith("uploads/"):
        return False
    return is_within(config.UPLOAD_DIR, absolute)


def audio_path_is_safe(relative_path: str) -> bool:
    absolute = static_relative_to_absolute(relative_path)
    if absolute is None:
        return False
    if not relative_path.replace("\\", "/").startswith("audio/"):
        return False
    return is_within(config.AUDIO_DIR, absolute)
