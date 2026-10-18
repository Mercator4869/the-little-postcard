"""图片校验与安全测试（计划书 §17.3）。"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import image_utils  # noqa: E402
from tests.conftest import make_image_bytes  # noqa: E402


class TestValidation:
    def test_jpeg_accepted(self):
        result = image_utils.validate_and_normalize(make_image_bytes(fmt="JPEG"))
        with Image.open(io.BytesIO(result)) as img:
            assert img.format == "JPEG"

    def test_png_accepted(self):
        result = image_utils.validate_and_normalize(make_image_bytes(fmt="PNG"))
        with Image.open(io.BytesIO(result)) as img:
            assert img.format == "JPEG"  # 统一归一化为 JPEG

    def test_rgba_flattened_to_white(self):
        result = image_utils.validate_and_normalize(
            make_image_bytes(fmt="PNG", mode="RGBA")
        )
        with Image.open(io.BytesIO(result)) as img:
            assert img.mode == "RGB"

    def test_empty_rejected(self):
        with pytest.raises(image_utils.ImageValidationError):
            image_utils.validate_and_normalize(b"")

    def test_garbage_rejected(self):
        with pytest.raises(image_utils.ImageValidationError):
            image_utils.validate_and_normalize(b"this is definitely not an image")

    def test_fake_extension_rejected(self):
        """伪装扩展名的非图片内容被拒绝（不能只看扩展名）。"""
        with pytest.raises(image_utils.ImageValidationError):
            image_utils.validate_and_normalize(b"GIF89a" + b"\x00" * 100)

    def test_oversized_bytes_rejected(self):
        payload = make_image_bytes(size=(32, 32))
        with pytest.raises(image_utils.ImageValidationError):
            image_utils.validate_and_normalize(payload, max_bytes=len(payload) - 1)

    def test_oversized_pixels_rejected(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 100)
        with pytest.raises(image_utils.ImageValidationError):
            image_utils.validate_and_normalize(make_image_bytes(size=(64, 64)))

    def test_resized_within_max_side(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_IMAGE_SIDE", 100)
        result = image_utils.validate_and_normalize(make_image_bytes(size=(400, 200)))
        with Image.open(io.BytesIO(result)) as img:
            assert max(img.size) <= 100

    def test_aspect_ratio_preserved(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_IMAGE_SIDE", 100)
        result = image_utils.validate_and_normalize(make_image_bytes(size=(400, 200)))
        with Image.open(io.BytesIO(result)) as img:
            width, height = img.size
            assert abs((width / height) - 2.0) < 0.05

    def test_metadata_stripped(self):
        buffer = io.BytesIO()
        image = Image.new("RGB", (64, 64), (10, 20, 30))
        exif = image.getexif()
        exif[0x010F] = "TestCamera"
        image.save(buffer, format="JPEG", exif=exif)

        result = image_utils.validate_and_normalize(buffer.getvalue())
        with Image.open(io.BytesIO(result)) as img:
            assert not img.getexif()


class TestNormalizeIdempotency:
    """规范化必须幂等，否则数据库 sha256 会与磁盘文件哈希漂移。"""

    def test_double_normalize_is_stable(self):
        once = image_utils.normalize_once(make_image_bytes(fmt="JPEG"))
        twice = image_utils.normalize_once(once)
        assert once == twice
        assert image_utils.sha256_of_bytes(once) == image_utils.sha256_of_bytes(twice)

    def test_triple_normalize_is_stable(self):
        once = image_utils.normalize_once(make_image_bytes(fmt="PNG"))
        twice = image_utils.normalize_once(once)
        thrice = image_utils.normalize_once(twice)
        assert once == twice == thrice

    def test_png_normalized_once_to_jpeg(self):
        result = image_utils.normalize_once(make_image_bytes(fmt="PNG"))
        with Image.open(io.BytesIO(result)) as img:
            assert img.format == "JPEG"
            assert img.mode == "RGB"

    def test_size_limit_enforced_on_already_normalized(self):
        """已规范化的内容也必须受体积上限约束（防绕过）。"""
        normalized = image_utils.normalize_once(make_image_bytes(fmt="JPEG"))
        with pytest.raises(image_utils.ImageValidationError):
            image_utils.normalize_once(normalized, max_bytes=len(normalized) - 1)

    def test_size_limit_enforced_on_raw(self):
        payload = make_image_bytes(fmt="JPEG")
        with pytest.raises(image_utils.ImageValidationError):
            image_utils.normalize_once(payload, max_bytes=len(payload) - 1)

    def test_truncated_jpeg_not_treated_as_normalized(self):
        """截断的 JPEG 不能被"已规范化"快捷路径放过。

        只看文件头时它很像一张合法的小尺寸 RGB JPEG，
        若不做 verify() 就会被原样返回，用户拿到的不是
        "图片已损坏"而是莫名其妙的兜底结果。

        用带噪声的大图，保证截断点一定落在压缩数据中间。
        """
        noise = Image.frombytes(
            "RGB", (256, 256), bytes((i * 7 + 13) % 256 for i in range(256 * 256 * 3))
        )
        buffer = io.BytesIO()
        noise.save(buffer, format="JPEG", quality=90)
        normalized = image_utils.normalize_once(buffer.getvalue())

        # 只保留前 60%：文件头完整，压缩数据被截断
        truncated = normalized[: int(len(normalized) * 0.6)]
        with Image.open(io.BytesIO(truncated)) as probe:
            # 前提：文件头仍可解析，说明"只看头部"确实会误判
            assert probe.format == "JPEG"

        assert image_utils._is_already_normalized(truncated) is False
        with pytest.raises(image_utils.ImageValidationError):
            image_utils.normalize_once(truncated)

    def test_valid_normalized_output_is_recognized(self):
        """自身产出的规范化字节必须被快捷路径认可（幂等前提）。"""
        normalized = image_utils.normalize_once(make_image_bytes(fmt="JPEG"))
        assert image_utils._is_already_normalized(normalized) is True

    def test_empty_rejected(self):
        with pytest.raises(image_utils.ImageValidationError):
            image_utils.normalize_once(b"")

    def test_large_image_gets_resized_then_stable(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_IMAGE_SIDE", 100)
        first = image_utils.normalize_once(make_image_bytes(size=(400, 400)))
        with Image.open(io.BytesIO(first)) as img:
            assert max(img.size) <= 100
        # 再规范化不应改变结果
        assert image_utils.normalize_once(first) == first


class TestPathSafety:
    def test_valid_upload_path(self):
        assert image_utils.static_relative_to_absolute("uploads/abc.jpg") is not None

    def test_rejects_absolute_path(self):
        assert image_utils.static_relative_to_absolute("C:/Windows/system32/x.jpg") is None

    def test_rejects_parent_traversal(self):
        assert image_utils.static_relative_to_absolute("uploads/../../secrets.txt") is None

    def test_rejects_wrong_prefix(self):
        assert image_utils.static_relative_to_absolute("css/style.css") is None

    def test_rejects_empty(self):
        assert image_utils.static_relative_to_absolute("") is None

    def test_rejects_leading_slash(self):
        assert image_utils.static_relative_to_absolute("/uploads/x.jpg") is None

    def test_rejects_backslash_traversal(self):
        assert image_utils.static_relative_to_absolute("uploads\\..\\..\\x.jpg") is None

    def test_rejects_empty_segment(self):
        assert image_utils.static_relative_to_absolute("uploads//x.jpg") is None

    def test_rejects_current_dir_segment(self):
        assert image_utils.static_relative_to_absolute("uploads/./x.jpg") is None

    def test_rejects_non_string(self):
        assert image_utils.static_relative_to_absolute(123) is None
        assert image_utils.static_relative_to_absolute(None) is None

    def test_accepts_backslash_separator(self):
        """Windows 风格分隔符归一化后仍应通过。"""
        assert image_utils.static_relative_to_absolute("uploads\\x.jpg") is not None

    def test_upload_path_is_safe(self):
        assert image_utils.upload_path_is_safe("uploads/x.jpg")
        assert not image_utils.upload_path_is_safe("comment_images/x.jpg")
        assert not image_utils.upload_path_is_safe("../uploads/x.jpg")

    def test_audio_path_is_safe(self):
        assert image_utils.audio_path_is_safe("audio/x.mp3")
        assert not image_utils.audio_path_is_safe("uploads/x.mp3")
        assert not image_utils.audio_path_is_safe("audio/../../../x.mp3")


class TestHashing:
    def test_same_content_same_hash(self):
        data = make_image_bytes()
        assert image_utils.sha256_of_bytes(data) == image_utils.sha256_of_bytes(data)

    def test_different_content_different_hash(self):
        a = image_utils.sha256_of_bytes(make_image_bytes(color=(1, 2, 3)))
        b = image_utils.sha256_of_bytes(make_image_bytes(color=(4, 5, 6)))
        assert a != b
