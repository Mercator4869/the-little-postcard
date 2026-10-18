"""混合检索：图片相似度 + 留言文本相似度 + 文字情感，按 RRF 融合。

为什么用 RRF（Reciprocal Rank Fusion）而不是原始余弦加权：
- 图片与文本来自两个不同模型，相似度尺度不可比；
- 205 条留言中 57 条的配图与评论素材逐字节相同，图片相似度会饱和到 1.0；
- RRF 只看名次，天然免疫尺度差异与饱和。

规则：
- 图片始终参与；文本 / 情感仅在查询文本足够长（≥ EMOTION_MIN_TEXT_LENGTH）
  且候选存在对应信号时参与；
- 无候选信号不惩罚、不补零；
- 完全并列取 item_id 最小者；
- 分数只在内部排序，不落库、不展示。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

import config
import db
import image_utils

logger = logging.getLogger(__name__)

POOL_COMMENT = "comment_image"
POOL_POST = "post"


@dataclass(frozen=True)
class Item:
    item_type: str
    item_id: int
    song_id: int
    image_path: str
    text: str | None
    image_vec: np.ndarray | None
    text_vec: np.ndarray | None
    valence: float | None
    emotion_label: str | None


@dataclass(frozen=True)
class MatchOutcome:
    item: Item
    fused_score: float
    used_text: bool
    used_emotion: bool


class PoolUnavailable(RuntimeError):
    """该检索池当前不可用（无模型 / 无有效向量）。"""


def _valid_vector(raw: str | None, model_name: str | None, expected_model: str):
    if not raw or model_name != expected_model:
        return None
    try:
        data = json.loads(raw)
        vector = np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.shape != (config.EMBEDDING_DIM,):
        return None
    if not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return (vector / norm).astype(np.float32)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1 起的名次；分数完全相同的候选共享同一名次（并列）。"""
    if len(scores) == 0:
        return np.zeros(0, dtype=np.int32)
    _unique, inverse = np.unique(-scores, return_inverse=True)
    return (inverse + 1).astype(np.int32)


class PoolIndex:
    """一个候选池的图片/文本/情绪索引。"""

    def __init__(
        self,
        items: list[Item],
        image_rows: list[int],
        image_matrix: np.ndarray,
        text_rows: list[int],
        text_matrix: np.ndarray,
    ) -> None:
        self.items = items
        self.ids = np.asarray([item.item_id for item in items], dtype=np.int64)
        self.image_rows = np.asarray(image_rows, dtype=np.int64)
        self.image_matrix = image_matrix
        self.text_rows = np.asarray(text_rows, dtype=np.int64)
        self.text_matrix = text_matrix
        valence = [item.valence if item.valence is not None else np.nan for item in items]
        self.valence = np.asarray(valence, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def empty(cls) -> "PoolIndex":
        return cls(
            [],
            [],
            np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32),
            [],
            np.zeros((0, config.TEXT_EMBEDDING_DIM), dtype=np.float32),
        )

    @property
    def is_empty(self) -> bool:
        return not self.items or self.image_matrix.size == 0

    @property
    def has_text(self) -> bool:
        return self.text_matrix.size > 0

    # ------------------------------------------------------------- 构建
    @classmethod
    def build(cls, pool: str, *, require_text: bool = False) -> "PoolIndex":
        with db.get_connection() as conn:
            if pool == POOL_COMMENT:
                rows = db.load_comment_index_rows(conn)
            else:
                rows = db.load_post_index_rows(conn)

        items: list[Item] = []
        image_rows: list[int] = []
        image_vecs: list[np.ndarray] = []
        text_rows: list[int] = []
        text_vecs: list[np.ndarray] = []
        skipped: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        for row in rows:
            item_id = int(row["item_id"])
            relative_path = str(row["image_local_path"])
            absolute = image_utils.static_relative_to_absolute(relative_path)
            if absolute is None or not absolute.exists():
                skip("图片文件不存在或路径非法")
                continue

            image_vec = _valid_vector(
                row["embedding_json"], row["embedding_model"], config.CLIP_MODEL_NAME
            )
            text_vec = _valid_vector(
                row["text_embedding_json"], row["text_model"], config.TEXT_MODEL_NAME
            )
            text = (row["note_text"] or "").strip() or None
            if require_text and text is None:
                # 最终输出必须是「图片 + 笔记」组合，没有配文的图不参与匹配
                skip("缺少配文")
                continue
            valence = row["valence"]
            if valence is not None:
                valence = float(valence)
                if not np.isfinite(valence):
                    valence = None
            # 候选文本太短时情绪不可靠：按中性处理，只保留图片/文本信号
            if text is not None and len(text) < config.EMOTION_MIN_TEXT_LENGTH:
                valence = None
            label = row["emotion_label"]

            position = len(items)
            items.append(
                Item(
                    item_type=pool,
                    item_id=item_id,
                    song_id=int(row["song_id"]),
                    image_path=relative_path,
                    text=text,
                    image_vec=image_vec,
                    text_vec=text_vec,
                    valence=valence,
                    emotion_label=label,
                )
            )
            if image_vec is not None:
                image_rows.append(position)
                image_vecs.append(image_vec)
            if text_vec is not None:
                text_rows.append(position)
                text_vecs.append(text_vec)

        if skipped:
            for reason, count in sorted(skipped.items()):
                logger.warning("跳过 %d 条 %s 索引：%s", count, pool, reason)

        image_matrix = (
            np.vstack(image_vecs).astype(np.float32)
            if image_vecs
            else np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        )
        text_matrix = (
            np.vstack(text_vecs).astype(np.float32)
            if text_vecs
            else np.zeros((0, config.TEXT_EMBEDDING_DIM), dtype=np.float32)
        )

        logger.info(
            "索引就绪：%s 共 %d 条（图片 %d，文本 %d，情绪 %d）",
            pool,
            len(items),
            len(image_rows),
            len(text_rows),
            int(np.sum(~np.isnan([item.valence if item.valence is not None else np.nan for item in items]))),
        )
        return cls(items, image_rows, image_matrix, text_rows, text_matrix)

    # ------------------------------------------------------------- 检索
    def search(
        self,
        image_vec: np.ndarray,
        text_vec: np.ndarray | None = None,
        valence: float | None = None,
    ) -> MatchOutcome | None:
        if self.is_empty:
            return None

        count = len(self.items)
        fused = np.zeros(count, dtype=np.float64)
        used_text = False
        used_emotion = False

        # 文本/情绪只允许在「图片相似度接近最佳」的候选间重排；
        # 否则一句「难过」就能把视觉上毫无关系的候选拉上来。
        allowed = np.ones(count, dtype=bool)
        if self.image_matrix.size:
            image_similarity = np.full(count, -np.inf, dtype=np.float32)
            image_similarity[self.image_rows] = (
                self.image_matrix @ image_vec.astype(np.float32)
            )
            image_ranks = _ranks(image_similarity[self.image_rows])
            fused[self.image_rows] += config.HYBRID_IMAGE_WEIGHT / (
                config.HYBRID_RRF_K + image_ranks
            )
            best_image = float(np.max(image_similarity[self.image_rows]))
            allowed = image_similarity >= (
                best_image - config.HYBRID_RERANK_IMAGE_MARGIN
            )

        if text_vec is not None and self.text_matrix.size:
            similarity = np.full(count, -np.inf, dtype=np.float32)
            similarity[self.text_rows] = self.text_matrix @ text_vec.astype(np.float32)
            eligible = allowed & np.isfinite(similarity)
            if np.any(eligible):
                indices = np.flatnonzero(eligible)
                ranks = _ranks(similarity[indices])
                fused[indices] += config.HYBRID_TEXT_WEIGHT / (
                    config.HYBRID_RRF_K + ranks
                )
                used_text = True

        if valence is not None:
            emotion = np.full(count, np.nan, dtype=np.float32)
            valid = ~np.isnan(self.valence)
            emotion[valid] = 1.0 - np.abs(self.valence[valid] - float(valence)) / 2.0
            eligible = allowed & np.isfinite(emotion)
            if np.any(eligible):
                indices = np.flatnonzero(eligible)
                ranks = _ranks(emotion[indices])
                fused[indices] += config.HYBRID_EMOTION_WEIGHT / (
                    config.HYBRID_RRF_K + ranks
                )
                used_emotion = True

        if not np.any(fused > 0):
            return None

        # 先按融合分降序，再按 item_id 升序 → 完全并列时 id 最小者在前
        order = np.lexsort((self.ids, -fused))
        best = int(order[0])
        if fused[best] <= 0:
            return None
        return MatchOutcome(
            item=self.items[best],
            fused_score=float(fused[best]),
            used_text=used_text,
            used_emotion=used_emotion,
        )


class HybridService:
    """把 CLIP、文本模型、情感模型与「图片 + 笔记」候选池串起来。"""

    def __init__(
        self,
        *,
        encoder,
        text_encoder,
        sentiment,
        comment_index: PoolIndex,
        fallback_song_id: int | None,
        load_failure_reason: str | None = None,
    ) -> None:
        self._encoder = encoder
        self._text_encoder = text_encoder
        self._sentiment = sentiment
        self._comment_index = comment_index
        self._fallback_song_id = fallback_song_id
        self._load_failure_reason = load_failure_reason

    # ---------------------------------------------------------- 能力
    @property
    def model_available(self) -> bool:
        return self._encoder is not None

    @property
    def text_available(self) -> bool:
        return self._text_encoder is not None

    @property
    def emotion_available(self) -> bool:
        return self._sentiment is not None

    @property
    def comment_index_size(self) -> int:
        return len(self._comment_index)

    def refresh_comment_index(self) -> None:
        """把刚写入数据库的明信片加入下一次检索，不需要重启服务。"""
        self._comment_index = PoolIndex.build(POOL_COMMENT, require_text=True)
        logger.info("评论图索引已刷新：%d 条", len(self._comment_index))

    @property
    def load_failure_reason(self) -> str | None:
        return self._load_failure_reason

    def encode_image(self, image) -> np.ndarray:
        if self._encoder is None:
            raise PoolUnavailable(self._load_failure_reason or "图片模型不可用")
        return self._encoder.encode_image(image)

    def encode_text(self, text: str) -> np.ndarray | None:
        if self._text_encoder is None or not text:
            return None
        try:
            return self._text_encoder.encode(text)
        except Exception as exc:  # noqa: BLE001
            logger.error("文本编码失败：%s", exc)
            return None

    def score_text(self, text: str) -> float | None:
        if self._sentiment is None or not text:
            return None
        try:
            valence, _label, _probability = self._sentiment.score(text)
            return float(valence)
        except Exception as exc:  # noqa: BLE001
            logger.error("情感打分失败：%s", exc)
            return None

    # ---------------------------------------------------------- 检索
    def match_song(
        self,
        image_vec: np.ndarray,
        text_vec: np.ndarray | None = None,
        valence: float | None = None,
    ) -> MatchOutcome | None:
        return self._comment_index.search(image_vec, text_vec, valence)

    def fallback_song_id(self) -> int | None:
        return self._fallback_song_id
