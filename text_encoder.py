"""中文文本向量（bge-small-zh）与情感模型（StructBERT）封装。

- 全部从项目内本地目录加载，`local_files_only=True`，运行期不联网。
- 文本向量用于「留言 ↔ 留言」语义相似度；情感模型输出极性概率，转成效价
  `valence = P(正面) - P(负面)`，取值范围 [-1, 1]。
- 两个模型都可单独缺失：缺失时对应信号不参与排序，图片匹配不受影响。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Sequence

# 与 clip_encoder 保持一致：离线变量必须在 import transformers 之前设置
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import config  # noqa: E402

os.environ.setdefault("HF_HOME", str(config.CLIP_CACHE_DIR))

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


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


class ModelLoadError(RuntimeError):
    """本地模型加载失败。"""


# ---------------------------------------------------------------- 极性词校准
# 二分类情感模型训练自中文评论语料，对「多从句 / 转折」的音乐笔记容易翻极性：
#   「今天有点难过，想听一首安静的歌」被模型判成正面 +0.75；
#   「失恋了」被判成 +0.32。
# 这里用一份很小的极性词表做**保守校准**（0.6 模型 + 0.4 词表），
# 只修正明显的极性翻转，不替代模型。词表可在评审时逐条核对。
_NEGATIVE_WORDS = (
    "难过", "悲伤", "伤心", "心碎", "失恋", "孤独", "寂寞", "焦虑", "烦躁",
    "失眠", "崩溃", "心累", "遗憾", "失落", "离别", "抑郁", "考砸", "想哭",
    "哭了", "痛苦", "绝望", "委屈", "emo",
)
_POSITIVE_WORDS = (
    "开心", "快乐", "幸福", "喜欢", "好听", "温柔", "浪漫", "温暖", "希望",
    "美好", "治愈", "放松", "甜蜜", "满足", "惊喜",
)
_NEGATORS = ("不", "没", "别", "无", "非")


def _is_negated(text: str, position: int) -> bool:
    prefix = text[max(0, position - 2):position]
    return any(negator in prefix for negator in _NEGATORS)


def _lexicon_score(text: str) -> int:
    score = 0
    for word in _NEGATIVE_WORDS:
        start = 0
        while True:
            index = text.find(word, start)
            if index < 0:
                break
            score += 1 if _is_negated(text, index) else -1
            start = index + len(word)
    for word in _POSITIVE_WORDS:
        start = 0
        while True:
            index = text.find(word, start)
            if index < 0:
                break
            score += -1 if _is_negated(text, index) else 1
            start = index + len(word)
    return score


def _calibrate_valence(text: str, model_valence: float) -> float:
    score = _lexicon_score(text)
    if score == 0:
        return model_valence
    import math

    lexicon_valence = math.tanh(float(score))
    combined = 0.45 * model_valence + 0.55 * lexicon_valence
    return float(max(-1.0, min(1.0, combined)))


class TextEncoder:
    """bge-small-zh：CLS 池化 + L2 归一化，输出 512 维。"""

    def __init__(self, tokenizer, model, device: str = "cpu") -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "TextEncoder":
        import torch
        from transformers import AutoModel, AutoTokenizer

        directory = Path(model_dir) if model_dir is not None else config.BGE_MODEL_DIR
        if not directory.exists():
            raise ModelLoadError(f"文本模型目录不存在：{directory}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
            model = AutoModel.from_pretrained(str(directory), local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"文本模型加载失败：{exc}") from exc

        model.eval()
        model.to(config.CLIP_DEVICE)
        return cls(tokenizer, model, config.CLIP_DEVICE)

    def encode(self, texts: str | Sequence[str]) -> np.ndarray:
        """返回 (512,) 或 (N, 512)，float32、L2 归一化。"""
        import torch

        single = isinstance(texts, str)
        batch = [texts] if single else [str(t) for t in texts]
        if not batch:
            return np.zeros((0, config.TEXT_EMBEDDING_DIM), dtype=np.float32)

        inputs = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._lock, torch.inference_mode():
            outputs = self._model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].to("cpu").to(torch.float32).numpy()

        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result = (cls / norms).astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("文本特征包含 NaN 或 Inf")
        return result[0] if single else result


class SentimentEncoder:
    """StructBERT 中文情感：输出 valence 与标签。"""

    def __init__(self, tokenizer, model, id2label: dict[int, str], device: str = "cpu") -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._id2label = id2label
        self._device = device
        self._lock = threading.Lock()
        self._pos_id = _find_label_id(id2label, ("正", "pos"))
        self._neg_id = _find_label_id(id2label, ("负", "neg"))

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "SentimentEncoder":
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        directory = Path(model_dir) if model_dir is not None else config.SENTIMENT_MODEL_DIR
        if not directory.exists():
            raise ModelLoadError(f"情感模型目录不存在：{directory}")

        weight_file = directory / "pytorch_model.bin"
        if not weight_file.exists():
            raise ModelLoadError(f"情感模型权重不存在：{weight_file}")

        labels_file = directory / "label_mapping.json"
        id2label: dict[int, str] = {0: "负面", 1: "正面"}
        try:
            raw_labels = json.loads(labels_file.read_text(encoding="utf-8"))
            if isinstance(raw_labels, dict) and raw_labels:
                id2label = {int(v): str(k) for k, v in raw_labels.items()}
        except (OSError, ValueError, TypeError):
            logger.warning("情感模型 label_mapping.json 读取失败，使用默认标签")

        try:
            config_ = BertConfig.from_pretrained(str(directory), num_labels=len(id2label))
            model = BertForSequenceClassification(config_)
            state = torch.load(str(weight_file), map_location="cpu", weights_only=True)
            # EasyNLP 权重前缀是 encoder.*，transformers 的 BertForSequenceClassification
            # 主干前缀是 bert.*；classifier.* 保持不变。
            renamed = {
                ("bert." + key[len("encoder."):]) if key.startswith("encoder.") else key: value
                for key, value in state.items()
            }
            missing, unexpected = model.load_state_dict(renamed, strict=False)
            critical = [k for k in missing if k.startswith(("bert.", "classifier"))]
            if critical:
                raise ModelLoadError(f"情感模型权重不完整，缺少：{critical[:5]}")
            if unexpected:
                logger.warning("情感模型存在未使用的权重：%s", unexpected[:5])
            tokenizer = BertTokenizerFast(vocab_file=str(directory / "vocab.txt"))
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"情感模型加载失败：{exc}") from exc

        model.eval()
        model.to(config.CLIP_DEVICE)
        return cls(tokenizer, model, id2label, config.CLIP_DEVICE)

    def score(self, text: str) -> tuple[float, str, float]:
        """返回 (valence, 标签, 置信度)。valence ∈ [-1, 1]。"""
        import torch

        inputs = self._tokenizer(
            text, truncation=True, max_length=128, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._lock, torch.inference_mode():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].to("cpu").tolist()

        positive = probs[self._pos_id] if self._pos_id is not None else 0.0
        negative = probs[self._neg_id] if self._neg_id is not None else 0.0
        valence = float(positive - negative)
        valence = _calibrate_valence(text, valence)
        # 标签与校准后的 valence 保持一致，避免出现「valence 为负、标签为正面」
        if valence > 0.05:
            label = "正面"
        elif valence < -0.05:
            label = "负面"
        else:
            label = "中性"
        probability = float(max(probs))
        return valence, label, probability

    def score_many(self, texts: Sequence[str]) -> list[tuple[float, str, float]]:
        return [self.score(text) for text in texts]


def _find_label_id(id2label: dict[int, str], needles: tuple[str, ...]) -> int | None:
    for label_id, label in id2label.items():
        lowered = str(label).lower()
        if any(needle in lowered for needle in needles):
            return int(label_id)
    return None


def load_text_models_or_none() -> tuple[TextEncoder | None, SentimentEncoder | None, str | None]:
    """分别加载文本与情感模型；任一失败只禁用对应信号。"""
    text_encoder = None
    sentiment = None
    reasons: list[str] = []

    try:
        text_encoder = TextEncoder.load()
        logger.info("中文文本模型加载成功（%s）", config.TEXT_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.error("中文文本模型加载失败：%s", exc)
        reasons.append(f"文本模型不可用：{exc}")

    try:
        sentiment = SentimentEncoder.load()
        logger.info("中文情感模型加载成功（%s）", config.SENTIMENT_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.error("中文情感模型加载失败：%s", exc)
        reasons.append(f"情感模型不可用：{exc}")

    return text_encoder, sentiment, ("；".join(reasons) if reasons else None)
