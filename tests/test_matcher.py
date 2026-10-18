"""检索逻辑测试（计划书 §17.1）。

验证核心规则：只取余弦相似度最高的评论图片，返回其所属歌曲；
无阈值、无歌曲级聚合、无 top2 差值判断。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from matcher import ImageRecord, MaterialIndex, MatchService  # noqa: E402


def unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(config.EMBEDDING_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeEncoder:
    """返回预设向量的假编码器，用于纯逻辑测试（不加载真实模型）。"""

    def __init__(self, vector: np.ndarray, *, fail: bool = False):
        self._vector = vector
        self._fail = fail

    def encode_image(self, image) -> np.ndarray:
        if self._fail:
            raise RuntimeError("模拟推理失败")
        return self._vector


def build_index(records: list[tuple[int, int, np.ndarray]]) -> MaterialIndex:
    """records: [(comment_image_id, song_id, vector)]"""
    vectors = np.vstack([v for _, _, v in records]).astype(np.float32)
    image_records = [
        ImageRecord(comment_image_id=cid, song_id=sid, image_local_path=f"comment_images/{cid}.jpg")
        for cid, sid, _ in records
    ]
    return MaterialIndex(vectors, image_records)


class TestBestMatch:
    def test_picks_highest_similarity(self):
        target = unit_vector(1)
        index = build_index([
            (1, 10, unit_vector(2)),
            (2, 20, target),          # 与查询完全一致
            (3, 30, unit_vector(3)),
        ])
        record = index.best_match(target)
        assert record is not None
        assert record.comment_image_id == 2
        assert record.song_id == 20

    def test_no_song_level_aggregation(self):
        """即使某歌曲多张图片整体更接近，也只按单张最高分决定。"""
        query = unit_vector(5)
        index = build_index([
            (1, 10, query),            # 单张完全命中 → 应胜出
            (2, 20, unit_vector(6)),
            (3, 20, unit_vector(7)),
            (4, 20, unit_vector(8)),
        ])
        record = index.best_match(query)
        assert record.song_id == 10

    def test_tie_break_by_smallest_image_id(self):
        """完全并列时取评论图片 ID 最小者，结果可复现。"""
        same = unit_vector(9)
        index = build_index([
            (7, 70, same),
            (3, 30, same),
            (5, 50, same),
        ])
        record = index.best_match(same)
        assert record.comment_image_id == 3

    def test_single_image_works(self):
        query = unit_vector(11)
        index = build_index([(1, 100, query)])
        record = index.best_match(query)
        assert record is not None and record.song_id == 100

    def test_empty_index_returns_none(self):
        index = MaterialIndex(np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32), [])
        assert index.best_match(unit_vector(1)) is None
        assert index.is_empty

    def test_low_similarity_still_returns_best(self):
        """匹配度不高也返回最高者，不换成兜底歌曲。"""
        query = unit_vector(100)
        other = unit_vector(200)
        # 构造一个与查询接近正交的向量
        orthogonal = other - np.dot(other, query) * query
        orthogonal = (orthogonal / np.linalg.norm(orthogonal)).astype(np.float32)

        index = build_index([(1, 42, orthogonal)])
        record = index.best_match(query)
        assert record is not None
        assert record.song_id == 42  # 仍然返回它，而非兜底


class TestMatchService:
    def test_model_unavailable_uses_fallback(self):
        index = build_index([(1, 10, unit_vector(1))])
        service = MatchService(
            encoder=None, load_failure_reason="模型未加载", index=index, fallback_song_id=999
        )
        result = service.match_image(None)
        assert result.degraded is True
        assert result.song_id == 999
        assert result.matched_image_id is None

    def test_empty_index_uses_fallback(self):
        empty = MaterialIndex(np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32), [])
        service = MatchService(
            encoder=FakeEncoder(unit_vector(1)), load_failure_reason=None,
            index=empty, fallback_song_id=999,
        )
        result = service.match_image(None)
        assert result.degraded is True
        assert result.song_id == 999

    def test_inference_failure_uses_fallback(self):
        index = build_index([(1, 10, unit_vector(1))])
        service = MatchService(
            encoder=FakeEncoder(unit_vector(1), fail=True), load_failure_reason=None,
            index=index, fallback_song_id=999,
        )
        result = service.match_image(None)
        assert result.degraded is True
        assert result.song_id == 999

    def test_fallback_song_missing_raises(self):
        """兜底歌曲自身不存在时，降级路径必须显式失败而非静默返回空。

        计划书 §2.4 把"兜底歌曲自身不存在"列为**不适用兜底**的情况；
        Web 层由 require_fallback 装饰器提前返回 503，
        这里保证直接调用服务时也不会悄悄给出一个错误的 song_id。
        """
        index = build_index([(1, 10, unit_vector(1))])
        service = MatchService(
            encoder=None, load_failure_reason="模型未加载",
            index=index, fallback_song_id=None,   # 兜底歌曲缺失
        )
        with pytest.raises(RuntimeError):
            service.match_image(None)

    def test_successful_match_not_degraded(self):
        query = unit_vector(77)
        index = build_index([(1, 5, query), (2, 6, unit_vector(78))])
        service = MatchService(
            encoder=FakeEncoder(query), load_failure_reason=None,
            index=index, fallback_song_id=999,
        )
        result = service.match_image(None)
        assert result.degraded is False
        assert result.song_id == 5
        assert result.matched_image_id == 1
        assert result.degrade_reason is None


class TestProhibitedLogic:
    """确认没有引入被计划书明文禁止的匹配逻辑。

    只检查可执行代码（剥离注释与文档字符串），
    因为 matcher.py 的文档里会列举这些被禁止的术语。
    """

    @staticmethod
    def _code_without_docs(module) -> str:
        import inspect
        import io as _io
        import tokenize

        source = inspect.getsource(module)
        pieces: list[str] = []
        reader = _io.StringIO(source).readline
        try:
            for token in tokenize.generate_tokens(reader):
                if token.type in (
                    tokenize.COMMENT,
                    tokenize.STRING,
                ):
                    continue
                pieces.append(token.string)
        except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
            # 不静默回退到原始源码：matcher 的文档会**列举**这些被禁止的术语
            # 来说明"我们不这么做"，一旦退化为全文搜索就会产生假失败，
            # 把排查方向完全带偏。这里直接让检查失败更诚实。
            raise AssertionError(
                f"无法剥离注释与字符串，禁止逻辑检查不可信：{exc}"
            ) from exc
        return " ".join(pieces)

    def test_no_threshold_logic_in_code(self):
        import matcher

        code = self._code_without_docs(matcher)
        for forbidden in ("top2", "threshold", "MIN_SIMILARITY", "confidence"):
            assert forbidden not in code, f"不应出现 {forbidden}"

    def test_result_has_no_score_field(self):
        from matcher import MatchResult

        field_names = set(MatchResult.__dataclass_fields__)
        assert "score" not in field_names
        assert "similarity" not in field_names
        assert field_names == {"song_id", "matched_image_id", "degraded", "degrade_reason"}

    def test_only_argmax_used_for_selection(self):
        """选取结果只通过 argmax，不做排序或过滤。"""
        import matcher

        code = self._code_without_docs(matcher)
        assert "argmax" in code
        for forbidden in ("argsort", "sort(", "topk"):
            assert forbidden not in code
