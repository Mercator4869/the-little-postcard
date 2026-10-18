"""CLIP 图片特征提取（计划书 §11.1）。

关键要求：
- 只使用图片特征，输出 (512,) float32、有限值、L2 范数约等于 1
- 模型与处理器均从本地缓存加载，local_files_only=True
- 离线环境变量必须在 import transformers 之前设置
- CPU 推理、model.eval()、torch.inference_mode()
- Web 运行时禁止自动下载模型
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---- 三重防线之一：离线环境变量必须在 import transformers 之前设置 ----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


# ---------------------------------------------------------------- 编码兜底
# 契约自检会输出中文。Windows 控制台若为 GBK(936)，print() 会抛
# UnicodeEncodeError，自检还没给出结论就崩了。这里在任何中文输出前切到 UTF-8。
def _force_utf8_stdout() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


_force_utf8_stdout()

import numpy as np  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

import config  # noqa: E402

# ---- 三重防线之零：HF 默认缓存在用户主目录（%USERPROFILE%\.cache\huggingface），
# 本机该位置不可写，transformers 导入时会打印
# "There was a problem when trying to write in your cache folder"。
# 统一指到项目内的模型缓存目录，既消除警告，也确保运行期不往项目外写文件。
os.environ.setdefault("HF_HOME", str(config.CLIP_CACHE_DIR))

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """模型无法从本地缓存加载。"""


class ClipImageEncoder:
    """统一封装。输出契约见 self.self_check()。"""

    def __init__(self, model, processor, device: str) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    # ------------------------------------------------------------- 加载
    @classmethod
    def load(cls, cache_dir: Path | None = None, device: str | None = None) -> "ClipImageEncoder":
        """从本地缓存加载模型；任何失败都抛 ModelLoadError。

        绝不触发联网下载。
        """
        import torch
        from transformers import CLIPModel, CLIPProcessor

        cache = Path(cache_dir) if cache_dir is not None else config.CLIP_CACHE_DIR
        target_device = device or config.CLIP_DEVICE

        if not cache.exists():
            raise ModelLoadError(f"模型缓存目录不存在：{cache}（请先执行首次缓存命令）")

        try:
            processor = CLIPProcessor.from_pretrained(
                config.CLIP_MODEL_NAME,
                cache_dir=str(cache),
                local_files_only=True,
            )
            # ---- 三重防线之二：API 参数强制只用本地文件 ----
            model = CLIPModel.from_pretrained(
                config.CLIP_MODEL_NAME,
                cache_dir=str(cache),
                local_files_only=True,
            )
        except Exception as exc:
            raise ModelLoadError(f"模型加载失败：{exc}") from exc

        model.eval()
        model.to(target_device)

        return cls(model, processor, target_device)

    # ------------------------------------------------------------- 推理
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """返回 (512,) float32、有限、L2 归一化的特征向量。"""
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)

        with torch.inference_mode():
            features = self._model.get_image_features(pixel_values=pixel_values)

        vector = features[0].detach().to("cpu").to(torch.float32).numpy()

        if vector.shape != (config.EMBEDDING_DIM,):
            raise RuntimeError(
                f"特征维度异常：期望 ({config.EMBEDDING_DIM},)，实际 {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("特征包含 NaN 或 Inf")

        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise RuntimeError("特征范数为零")

        return (vector / norm).astype(np.float32)

    def encode_image_bytes(self, data: bytes) -> np.ndarray:
        import io

        with Image.open(io.BytesIO(data)) as img:
            return self.encode_image(img.convert("RGB"))

    # ------------------------------------------------------------- 自检
    def self_check(self) -> dict[str, object]:
        """用一张真实图片验证输出契约，返回实测数值。"""
        probe = Image.new("RGB", (224, 224), (128, 128, 128))
        vector = self.encode_image(probe)
        return {
            "shape": tuple(vector.shape),
            "dtype": str(vector.dtype),
            "all_finite": bool(np.all(np.isfinite(vector))),
            "l2_norm": float(np.linalg.norm(vector)),
            "device": self._device,
            "model_name": config.CLIP_MODEL_NAME,
        }


def load_encoder_or_none() -> tuple[ClipImageEncoder | None, str | None]:
    """按 §11.2 启动状态机加载一次，返回 (encoder, 失败原因)。

    MODEL_DOWNGRADE=True 时不实例化、不检查缓存、不推理。
    失败时记录原因，Web 继续启动，后续走兜底。
    """
    if config.MODEL_DOWNGRADE:
        logger.warning("MODEL_DOWNGRADE=True：跳过 CLIP 加载，全部请求走兜底歌曲")
        return None, "MODEL_DOWNGRADE=True"

    try:
        encoder = ClipImageEncoder.load()
        logger.info("CLIP 模型加载成功（device=%s）", config.CLIP_DEVICE)
        return encoder, None
    except ModelLoadError as exc:
        logger.error("CLIP 模型加载失败：%s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.exception("CLIP 模型加载出现未预期错误")
        return None, str(exc)


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 模型加载失败（缓存缺失、文件损坏等）时给出可操作的提示，
    # 而不是抛一段裸堆栈——这一步正是排查离线问题的地方。
    try:
        encoder = ClipImageEncoder.load()
    except ModelLoadError as exc:
        print()
        print("模型加载失败：", exc)
        print()
        print("可能原因与对策：")
        print("  1. 尚未下载模型缓存 —— 见 README 第 3 步（唯一一次联网）")
        print(f"     期望缓存目录：{config.CLIP_CACHE_DIR}")
        print("  2. 缓存目录存在但不完整 —— 删除后重新下载")
        print("  3. 误开了离线模式但缓存未就绪 —— 首次下载时不要设置")
        print("     HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE")
        print()
        sys.exit(2)

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        # 与 preprocess.py 保持一致：相对路径按项目目录解析，不依赖当前工作目录。
        # 否则从别处执行会报"无法读取图片"，而文件其实存在，容易误判为图片损坏。
        if not image_path.is_absolute():
            image_path = config.BASE_DIR / image_path
        try:
            with Image.open(image_path) as img:
                vec = encoder.encode_image(img.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            print(f"无法读取图片 {image_path}：{exc}")
            sys.exit(2)
        print(f"图片：{image_path}")
        print(f"shape={vec.shape} dtype={vec.dtype} norm={np.linalg.norm(vec):.6f}")
        print(f"前 5 维：{vec[:5]}")
    else:
        report = encoder.self_check()
        print("契约自检：")
        for key, value in report.items():
            print(f"  {key} = {value}")

        ok = (
            report["shape"] == (config.EMBEDDING_DIM,)
            and report["dtype"] == "float32"
            and report["all_finite"]
            and abs(float(report["l2_norm"]) - 1.0) < 1e-5
        )
        print("结果：", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
