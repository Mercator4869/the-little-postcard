"""共享测试夹具。"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

# 让测试能 import 项目根目录下的模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_image_bytes(
    *,
    size: tuple[int, int] = (64, 64),
    mode: str = "RGB",
    fmt: str = "JPEG",
    color=(120, 160, 200),
) -> bytes:
    """生成一张真实图片的字节流（不是占位空文件）。"""
    if mode == "RGBA":
        image = Image.new("RGBA", size, (*color, 128))
    else:
        image = Image.new(mode, size, color)

    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image_bytes(fmt="JPEG")


@pytest.fixture
def png_bytes() -> bytes:
    return make_image_bytes(fmt="PNG", mode="RGBA")


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """把数据库指向临时文件，避免污染真实 data/music.db。"""
    import config
    import db

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    db.init_db()
    return db_path
