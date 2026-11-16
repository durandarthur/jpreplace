"""Fast tests that need no GPU, models or display."""

from __future__ import annotations

import numpy as np
import pytest

from jpreplace import win32
from jpreplace.layout import group_lines, has_japanese
from jpreplace.ocr import Line, _fix_long_vowel
from jpreplace.translate import Backend, Translator, _add_context, _Unit, build_prompt, split_sentences


def line(x0: float, y0: float, x1: float, y1: float, text: str) -> Line:
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    return Line(quad, text, 1.0, vertical=(y1 - y0) >= 1.5 * (x1 - x0))


def test_wrapped_paragraph_is_one_block():
    lines = [
        line(50, 100, 850, 122, "東京都は十六日、都営地下鉄の一部路線で金曜日の深夜運行を試験的に実施すると"),
        line(50, 130, 850, 152, "発表した。期間は来年一月から三か月間で、利用状況や安全面への影響を検証"),
        line(50, 160, 200, 182, "する。"),
    ]
    blocks = group_lines(lines)
    assert len(blocks) == 1
    assert blocks[0].text.startswith("東京都は") and blocks[0].text.endswith("する。")


def test_list_items_stay_separate():
    lines = [
        line(50, 100, 100, 122, "概要"),
        line(50, 132, 140, 154, "主要技術"),
        line(50, 164, 190, 186, "在来線との直通"),
    ]
    assert [b.text for b in group_lines(lines)] == ["概要", "主要技術", "在来線との直通"]


def test_menu_items_on_a_row_stay_separate():
    lines = [line(20, 10, 80, 34, "ファイル"), line(100, 10, 145, 34, "編集"), line(165, 10, 210, 34, "表示")]
    assert len(group_lines(lines)) == 3


def test_vertical_columns_read_right_to_left():
    lines = [
        line(100, 50, 135, 300, "聞いてないぞ！"),
        line(140, 50, 175, 280, "そんな話は"),
        line(180, 50, 215, 300, "なんだって！？"),
    ]
    blocks = group_lines(lines)
    assert len(blocks) == 1 and blocks[0].vertical
    assert blocks[0].text == "なんだって！？そんな話は聞いてないぞ！"


def test_has_japanese():
    assert has_japanese("設定") and has_japanese("ログイン") and has_japanese("ｶﾀｶﾅ")
    assert not has_japanese("Settings 123")


def test_long_vowel_fix():
    assert _fix_long_vowel("オ-ト") == "オート"
    assert _fix_long_vowel("セ-ブ") == "セーブ"
    assert _fix_long_vowel("A-B 10-20") == "A-B 10-20"


def test_prompt_format():
    prompt = build_prompt("会議です。", "文脈")
    assert prompt == (
        "<|plamo:tag|>dataset<|plamo:msg|>translation"
        "<|plamo:tag|>context<|plamo:msg|>文脈"
        "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>会議です。"
        "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
    )
    assert "context" not in build_prompt("会議です。")


def test_split_sentences():
    short = "これは短い文です。"
    assert split_sentences(short) == [short]
    long = "一つ目の文はここで終わります。" * 8
    chunks = split_sentences(long)
    assert len(chunks) > 1 and "".join(chunks) == long
    assert all(c.endswith("。") for c in chunks)
    quoted = "彼は「行くぞ！」と言った。" * 8
    assert all(not c.startswith("」") for c in split_sentences(quoted))


def test_context_goes_to_labels_and_later_sentences():
    texts = ["保存", "長い" * 60 + "。" + "次の文です。" * 10]
    units = [_Unit(i, chunk) for i, text in enumerate(texts) for chunk in split_sentences(text)]
    _add_context(units, texts)
    assert units[0].context  # a short label sees its neighbours
    assert units[1].context == ""  # a long opening chunk needs none
    assert units[2].context and units[2].context in units[1].text  # the sentence before


def test_backend_choice_and_cache():
    translator = Translator([Backend("http://quality", 4, max_chars=100), Backend("http://fast", 4)])
    try:
        assert translator.pick_backend(50) == 0
        assert translator.pick_backend(500) == 1
        assert translator.cached("未訳") is None
        translator._remember("既訳", "done")
        assert translator.cached("既訳") == "done"
        assert translator.translate(["既訳"]) == ["done"]  # served without any server
    finally:
        translator.close()


@pytest.mark.parametrize(
    "spec, expected",
    [("ctrl+alt+t", (0x3, ord("T"))), ("Ctrl + Shift + F5", (0x6, 0x74)), ("alt+space", (0x1, 0x20))],
)
def test_parse_hotkey(spec, expected):
    assert win32.parse_hotkey(spec) == expected


def test_parse_hotkey_rejects_nonsense():
    with pytest.raises(ValueError):
        win32.parse_hotkey("ctrl+alt")
    with pytest.raises(ValueError):
        win32.parse_hotkey("ctrl+banana")


def test_unpunctuated_text_is_still_bounded():
    text = "あ" * 100 + "、" + "い" * 300
    chunks = split_sentences(text)
    assert "".join(chunks) == text
    assert max(map(len, chunks)) <= 160
