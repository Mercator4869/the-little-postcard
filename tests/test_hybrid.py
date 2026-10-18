"""混合检索（RRF）单元测试：不加载任何真实模型。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from hybrid import Item, PoolIndex  # noqa: E402


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _item(item_id: int, *, valence: float | None = None, text: str | None = "x") -> Item:
    return Item(
        item_type="post",
        item_id=item_id,
        song_id=item_id * 10,
        image_path=f"uploads/{item_id}.jpg",
        text=text,
        image_vec=None,
        text_vec=None,
        valence=valence,
        emotion_label=None,
    )


def _pad_image(vector: np.ndarray) -> np.ndarray:
    out = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    out[: len(vector)] = vector
    return out


def _pad_text(vector: np.ndarray) -> np.ndarray:
    out = np.zeros(config.TEXT_EMBEDDING_DIM, dtype=np.float32)
    out[: len(vector)] = vector
    return out


def _qi(*values: float) -> np.ndarray:
    return _pad_image(_unit(*values))


def _qt(*values: float) -> np.ndarray:
    return _pad_text(_unit(*values))


def _index(items, image_rows, image_matrix, text_rows=(), text_matrix=None):
    shape = np.asarray(image_matrix, dtype=np.float32).reshape(-1, config.EMBEDDING_DIM)
    return PoolIndex(
        items,
        list(image_rows),
        shape,
        list(text_rows),
        (
            np.asarray(text_matrix, dtype=np.float32)
            if text_matrix is not None
            else np.zeros((0, config.TEXT_EMBEDDING_DIM), dtype=np.float32)
        ).reshape(-1, config.TEXT_EMBEDDING_DIM),
    )


class TestImageRanking:
    def test_image_only_returns_best(self):
        items = [_item(1), _item(2), _item(3)]
        matrix = np.vstack(
            [_pad_image(_unit(1, 0, 0)), _pad_image(_unit(0, 1, 0)), _pad_image(_unit(0, 0, 1))]
        )
        index = _index(items, [0, 1, 2], matrix)
        outcome = index.search(_qi(0, 1, 0))
        assert outcome is not None
        assert outcome.item.item_id == 2
        assert outcome.used_text is False
        assert outcome.used_emotion is False

    def test_tie_breaks_by_smallest_id(self):
        items = [_item(5), _item(3)]
        matrix = np.vstack([_pad_image(_unit(1, 0, 0)), _pad_image(_unit(1, 0, 0))])
        index = _index(items, [0, 1], matrix)
        outcome = index.search(_qi(1, 0, 0))
        assert outcome is not None
        assert outcome.item.item_id == 3

    def test_empty_index_returns_none(self):
        index = PoolIndex.empty()
        assert index.is_empty
        assert index.search(_qi(1, 0, 0)) is None


class TestTextAndEmotionTerms:
    def test_text_term_changes_score(self):
        items = [_item(1), _item(2)]
        matrix = np.vstack([_pad_image(_unit(1, 0, 0)), _pad_image(_unit(1, 0, 0))])
        text_matrix = np.vstack([_pad_text(_unit(1, 0)), _pad_text(_unit(-1, 0))])
        index = _index(items, [0, 1], matrix, [0, 1], text_matrix)

        image_only = index.search(_qi(1, 0, 0))
        with_text = index.search(_qi(1, 0, 0), _qt(1, 0), None)

        assert image_only is not None and with_text is not None
        assert with_text.used_text is True
        assert with_text.fused_score != image_only.fused_score

    def test_emotion_term_changes_score(self):
        items = [_item(1, valence=0.8), _item(2, valence=-0.8)]
        matrix = np.vstack([_pad_image(_unit(1, 0)), _pad_image(_unit(1, 0))])
        index = _index(items, [0, 1], matrix)

        without = index.search(_qi(1, 0), None, None)
        with_emotion = index.search(_qi(1, 0), None, -0.8)

        assert without is not None and with_emotion is not None
        assert with_emotion.used_emotion is True
        assert with_emotion.fused_score != without.fused_score

    def test_missing_text_does_not_break(self):
        items = [_item(1, text=None), _item(2, text=None)]
        matrix = np.vstack([_pad_image(_unit(1, 0)), _pad_image(_unit(0, 1))])
        index = _index(items, [0, 1], matrix)
        outcome = index.search(_qi(0, 1), _qt(1, 0), None)
        assert outcome is not None
        assert outcome.item.item_id == 2
        assert outcome.used_text is False

    def test_missing_valence_does_not_break(self):
        items = [_item(1, valence=None), _item(2, valence=None)]
        matrix = np.vstack([_pad_image(_unit(1, 0)), _pad_image(_unit(0, 1))])
        index = _index(items, [0, 1], matrix)
        outcome = index.search(_qi(0, 1), None, 0.8)
        assert outcome is not None
        assert outcome.used_emotion is False
