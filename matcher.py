"""检索匹配（计划书 §2、§11.3、§11.4）。

唯一匹配规则：
    用户上传图片 → 归一化特征 → 与全部有效评论图片向量算余弦相似度
    → 取匹配度最高的那一张 → 返回它所属歌曲。

明确禁止（§2.2）：图片质量/情绪评分、歌曲级聚合、top2_mean、
最低匹配阈值、Top1-Top2 差值判断、低置信度分类、随机推荐、
LLM 二次判断、标签加权、歌词/标题/音频参与匹配。

余弦相似度仅作内部排序依据：不落库、不展示百分比、不解释为概率。
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
import db
import image_utils

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRecord:
    """参与检索的一条评论图片记录。"""

    comment_image_id: int
    song_id: int
    image_local_path: str


@dataclass(frozen=True)
class MatchResult:
    """检索结果。matched_image_id 在降级时为 None。"""

    song_id: int
    matched_image_id: int | None
    degraded: bool
    degrade_reason: str | None


class MaterialIndex:
    """内存中的素材向量矩阵（N × 512 float32）。"""

    def __init__(self, matrix: np.ndarray, records: list[ImageRecord]) -> None:
        # 按评论图片 ID 升序排列，使「完全并列时取 ID 最小者」这一
        # §2.3 要求可复现。best_match 依赖 argmax 返回第一个最大值，
        # 因此顺序必须在**构造时**就确定下来，不能指望调用方按序传入
        # （直接从数据库读出的顺序、或测试里手工构造的顺序都未必有序）。
        if records:
            order = sorted(range(len(records)), key=lambda i: records[i].comment_image_id)
            self.records = [records[i] for i in order]
            if matrix is not None and matrix.size:
                self.matrix = matrix[order]
            else:
                self.matrix = matrix
        else:
            self.records = records
            self.matrix = matrix

    def __len__(self) -> int:
        return len(self.records)

    @property
    def is_empty(self) -> bool:
        return len(self.records) == 0 or self.matrix.size == 0

    # ------------------------------------------------------------- 构建
    @classmethod
    def build(cls) -> "MaterialIndex":
        """从数据库加载并校验全部有效向量。

        每条向量检查（§11.3）：JSON 可解析、维度 512、无 NaN/Inf、
        范数 > 0、模型标识一致、对应文件存在且路径合法。
        非法记录跳过并记录原因，不影响其他记录。
        """
        records: list[ImageRecord] = []
        vectors: list[np.ndarray] = []

        skipped: dict[str, int] = {}

        def _skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        try:
            with db.get_connection() as conn:
                rows = db.load_comment_image_rows(conn)
        except Exception as exc:
            logger.error("加载素材向量失败：%s", exc)
            return cls(np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32), [])

        for row in rows:
            image_id = int(row["id"])
            song_id = int(row["song_id"])
            relative_path = row["image_local_path"]

            # 模型标识必须一致
            if row["embedding_model"] != config.CLIP_MODEL_NAME:
                _skip("embedding_model 不一致")
                continue

            # 路径合法且文件存在
            absolute = image_utils.static_relative_to_absolute(relative_path)
            if absolute is None or not absolute.exists():
                _skip("图片文件不存在或路径非法")
                continue

            # JSON 可解析（JSONDecodeError 是 ValueError 的子类）
            try:
                raw = json.loads(row["embedding_json"])
            except (TypeError, ValueError):
                _skip("embedding_json 无法解析")
                continue

            try:
                vector = np.asarray(raw, dtype=np.float32)
            except (TypeError, ValueError):
                _skip("embedding_json 无法转为数值数组")
                continue

            # 维度、有限值、范数
            if vector.shape != (config.EMBEDDING_DIM,):
                _skip("维度不是 512")
                continue
            if not np.all(np.isfinite(vector)):
                _skip("包含 NaN 或 Inf")
                continue

            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                _skip("范数为零")
                continue

            # 合法向量重新归一化
            vectors.append((vector / norm).astype(np.float32))
            records.append(
                ImageRecord(
                    comment_image_id=image_id,
                    song_id=song_id,
                    image_local_path=relative_path,
                )
            )

        if skipped:
            for reason, count in sorted(skipped.items()):
                logger.warning("跳过 %d 条素材向量：%s", count, reason)

        if not vectors:
            logger.warning("素材库没有可用于匹配的有效向量")
            return cls(np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32), [])

        matrix = np.vstack(vectors).astype(np.float32)
        logger.info("素材索引就绪：%d 条有效向量", len(records))
        return cls(matrix, records)

    # ------------------------------------------------------------- 检索
    def best_match(self, query: np.ndarray) -> ImageRecord | None:
        """返回匹配度最高的评论图片记录。

        完全并列时取评论图片 ID 最小者（§2.3）：构造时已按 ID 升序排列，
        而 np.argmax 返回第一个最大值，行为因此可复现。
        """
        if self.is_empty:
            return None

        similarities = self.matrix @ query.astype(np.float32)
        if similarities.size == 0:
            return None

        best_index = int(np.argmax(similarities))
        return self.records[best_index]


# ------------------------------------------------------------------ 服务
class MatchService:
    """把编码器、素材索引、兜底歌曲串在一起。"""

    def __init__(
        self,
        encoder,
        load_failure_reason: str | None,
        index: MaterialIndex,
        fallback_song_id: int | None,
    ) -> None:
        self._encoder = encoder
        self._load_failure_reason = load_failure_reason
        self._index = index
        self._fallback_song_id = fallback_song_id

    @property
    def model_available(self) -> bool:
        return self._encoder is not None

    @property
    def index_size(self) -> int:
        return len(self._index)

    def rebuild_index(self) -> None:
        """素材导入或重建后刷新内存索引（需重启 Web 或调用此方法）。"""
        self._index = MaterialIndex.build()

    def fallback_result(self, reason: str) -> MatchResult:
        if self._fallback_song_id is None:
            raise RuntimeError("兜底歌曲不存在")
        return MatchResult(
            song_id=self._fallback_song_id,
            matched_image_id=None,
            degraded=True,
            degrade_reason=reason,
        )

    def match_image(self, image) -> MatchResult:
        """对一张已校验的图片执行检索。

        仅以下情况使用兜底（§2.4）：
        - 人工开启模型降级模式
        - 模型无法加载
        - 有效图片的推理过程失败
        - 素材库没有可用于匹配的有效向量
        """
        # 模型不可用
        if self._encoder is None:
            return self.fallback_result(self._load_failure_reason or "模型不可用")

        # 无有效向量
        if self._index.is_empty:
            return self.fallback_result("素材库没有有效的评论图片向量")

        # 推理失败
        try:
            query = self._encoder.encode_image(image)
        except Exception as exc:
            logger.error("图片推理失败：%s", exc)
            return self.fallback_result(f"图片推理失败：{exc}")

        record = self._index.best_match(query)
        if record is None:
            return self.fallback_result("素材库没有有效的评论图片向量")

        return MatchResult(
            song_id=record.song_id,
            matched_image_id=record.comment_image_id,
            degraded=False,
            degrade_reason=None,
        )


def build_match_service(encoder, load_failure_reason: str | None) -> MatchService:
    """启动时构建服务：加载兜底歌曲 ID 与素材索引。"""
    fallback_song_id: int | None = None
    try:
        with db.get_connection() as conn:
            row = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
            if row is not None:
                fallback_song_id = int(row["id"])
            else:
                logger.error("兜底歌曲不存在：song_key=%s", config.FALLBACK_SONG_KEY)
    except Exception as exc:
        logger.error("读取兜底歌曲失败：%s", exc)

    index = MaterialIndex.build()
    return MatchService(encoder, load_failure_reason, index, fallback_song_id)
