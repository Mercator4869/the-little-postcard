"""情感极性校准（不加载模型）：词表校准的边界行为。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text_encoder import _calibrate_valence, _lexicon_score  # noqa: E402


def test_negative_lexicon():
    assert _lexicon_score("今天真的很难过") == -1
    assert _lexicon_score("失恋了") == -1


def test_negation_flips_polarity():
    # 「不难过」「没有失眠」是轻微正向，不应记为负面
    assert _lexicon_score("不难过") == 1
    assert _lexicon_score("没有失眠") == 1


def test_positive_lexicon():
    assert _lexicon_score("好听到哭") > 0
    assert _lexicon_score("今天很温柔") > 0


def test_calibration_flips_model_error():
    # 模型对「多从句」容易判成 +0.75，词表把它拉回负向
    assert _calibrate_valence("今天有点难过，想听一首安静的歌", 0.75) < 0
    assert _calibrate_valence("失恋了", 0.32) < 0


def test_calibration_keeps_positive_when_lexicon_agrees():
    assert _calibrate_valence("好听到哭", 0.89) > 0.5


def test_calibration_untouched_without_lexicon_hit():
    assert _calibrate_valence("今天天气真好", 0.8) == 0.8
