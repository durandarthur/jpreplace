"""Render synthetic Japanese "screenshots" for testing and benchmarking.

Usage: python tests/make_samples.py [out_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen

ARTICLE = [
    ("見出し", "東京の地下鉄、深夜運行の実証実験を開始へ"),
    ("本文", "東京都は十六日、都営地下鉄の一部路線で金曜日の深夜運行を試験的に実施すると発表した。"
     "期間は来年一月から三か月間で、利用状況や安全面への影響を検証する。"),
    ("本文", "対象となるのは大江戸線と浅草線の二路線。終電の時刻をおよそ一時間半繰り下げ、"
     "午前二時ごろまで運行する計画だ。担当者は「夜間の経済活動を後押ししたい」と話している。"),
    ("本文", "一方で、保守点検の時間が短くなることへの懸念もあり、都は作業体制の見直しを進める方針だ。"),
]

MENU = ["ファイル", "編集", "表示", "履歴", "ブックマーク", "ツール", "ヘルプ"]
BUTTONS = ["ログイン", "新規登録", "お問い合わせ", "設定を保存する", "キャンセル"]

DIALOG = ("ユキ", "……本当に、行ってしまうの？\nあの約束、まだ覚えているよね。\n私はずっと、ここで待っているから。")

MANGA = [
    "なんだって！？\nそんな話は\n聞いてないぞ！",
    "落ち着いて。\nまずは状況を\n整理しましょう。",
    "時間がないんだ。\n今すぐ出発する。",
]


def _font(family: str, px: int, bold: bool = False) -> QFont:
    f = QFont(family)
    f.setPixelSize(px)
    f.setBold(bold)
    return f


def webpage(w: int = 1280, h: int = 720) -> QImage:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor("#ffffff"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.fillRect(0, 0, w, 44, QColor("#2b3a55"))
    p.setPen(QColor("#ffffff"))
    p.setFont(_font("Yu Gothic UI", 16))
    x = 24
    for item in MENU:
        p.drawText(QPointF(x, 28), item)
        x += p.fontMetrics().horizontalAdvance(item) + 36
    y = 80
    for kind, text in ARTICLE:
        if kind == "見出し":
            p.setFont(_font("Yu Gothic", 30, True))
            p.setPen(QColor("#111111"))
        else:
            p.setFont(_font("Meiryo", 18))
            p.setPen(QColor("#333333"))
        rect = QRectF(60, y, 820, 400)
        flags = Qt.TextFlag.TextWrapAnywhere | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        used = p.boundingRect(rect, flags, text)
        p.drawText(rect, flags, text)
        y += used.height() + 28
    bx = 940
    by = 90
    p.setFont(_font("Yu Gothic UI", 16, True))
    for i, label in enumerate(BUTTONS):
        r = QRectF(bx, by + i * 64, 260, 44)
        p.fillRect(r, QColor("#e8eefc") if i % 2 else QColor("#1a73e8"))
        p.setPen(QColor("#1a3c8a") if i % 2 else QColor("#ffffff"))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, label)
    p.end()
    return img


def game(w: int = 1280, h: int = 720) -> QImage:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    sky = QLinearGradient(0, 0, 0, h)
    sky.setColorAt(0, QColor("#27375e"))
    sky.setColorAt(0.6, QColor("#c9778b"))
    sky.setColorAt(1, QColor("#f2c48d"))
    p.fillRect(0, 0, w, h, sky)
    for i in range(14):
        p.fillRect(i * 97 - 20, h - 300 + (i * 53) % 90, 70, 300, QColor(20, 24, 40, 200))
    box = QRectF(80, h - 230, w - 160, 190)
    p.setBrush(QColor(10, 14, 30, 215))
    p.setPen(QColor(255, 255, 255, 180))
    p.drawRoundedRect(box, 14, 14)
    p.setPen(QColor("#ffd97a"))
    p.setFont(_font("Yu Gothic", 26, True))
    p.drawText(QPointF(box.left() + 30, box.top() + 44), DIALOG[0])
    p.setPen(QColor("#ffffff"))
    p.setFont(_font("Yu Mincho", 28))
    for i, line in enumerate(DIALOG[1].split("\n")):
        p.drawText(QPointF(box.left() + 30, box.top() + 88 + i * 40), line)
    p.setFont(_font("Yu Gothic UI", 15))
    p.setPen(QColor(255, 255, 255, 220))
    for i, label in enumerate(["セーブ", "ロード", "オート", "スキップ", "ログ"]):
        p.drawText(QPointF(w - 520 + i * 90, h - 14), label)
    p.end()
    return img


def manga(w: int = 900, h: int = 700) -> QImage:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor("#f4f4f4"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setPen(QColor("#000000"))
    p.setFont(_font("Yu Gothic", 26, True))
    step = 34
    for b, text in enumerate(MANGA):
        cx, cy = 180 + b * 270, 200 + (b % 2) * 180
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QPointF(cx, cy), 120, 170)
        cols = text.split("\n")
        x = cx + (len(cols) - 1) * step / 2 - 13
        for col in cols:
            y = cy - len(col) * step / 2 + 26
            for ch in col:
                p.drawText(QPointF(x, y), ch)
                y += step
            x -= step + 6
    p.end()
    return img


HATCHED = [
    ("diagonal", "ここから先は危険だ。", False),
    ("dots", "静かな夜だった。", True),
    ("cross", "誰もいないはずの部屋", False),
    ("stripes", "本日のおすすめ商品", False),
]


def _pattern(p: QPainter, kind: str, rect: QRectF) -> None:
    p.save()
    p.setClipRect(rect)
    p.fillRect(rect, QColor("#ffffff") if kind != "stripes" else QColor("#fdf3d8"))
    p.setPen(QPen(QColor("#202020"), 1.6))
    x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
    if kind in ("diagonal", "cross"):
        span = rect.width() + rect.height()
        for d in range(0, int(span), 7):
            p.drawLine(QPointF(x0 + d, y0), QPointF(x0 + d - rect.height(), y1))
            if kind == "cross":
                p.drawLine(QPointF(x0 + d - rect.height(), y0), QPointF(x0 + d, y1))
    elif kind == "dots":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#303030"))
        for row, y in enumerate(range(int(y0), int(y1) + 8, 8)):
            for x in range(int(x0) + (row % 2) * 4, int(x1) + 8, 8):
                p.drawEllipse(QPointF(x, y), 1.8, 1.8)
    else:
        for y in range(int(y0), int(y1), 12):
            p.fillRect(QRectF(x0, y, rect.width(), 6), QColor("#f3dfae"))
    p.restore()


def hatched(w: int = 1000, h: int = 720, with_text: bool = True) -> QImage:
    """Text on manga screentone and other repeating patterns."""
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor("#ffffff"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    font = _font("Yu Gothic", 30, True)
    for i, (kind, text, outlined) in enumerate(HATCHED):
        rect = QRectF(30 + (i % 2) * 485, 30 + (i // 2) * 345, 455, 315)
        _pattern(p, kind, rect)
        p.setPen(QPen(QColor("#000000"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)
        if not with_text:
            continue
        path = QPainterPath()
        path.addText(QPointF(rect.left() + 50, rect.top() + 150), font, text)
        if outlined:
            outline = QPen(QColor("#ffffff"), 7)
            outline.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.strokePath(path, outline)
        p.fillPath(path, QColor("#111111"))
    p.end()
    return img


SAMPLES = {"webpage": webpage, "game": game, "manga": manga, "hatched": hatched}


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/samples")
    out.mkdir(parents=True, exist_ok=True)
    app = QGuiApplication.instance() or QGuiApplication(["-platform", "offscreen"])
    for name, fn in SAMPLES.items():
        path = out / f"{name}.png"
        fn().save(str(path))
        print(path)
    del app


if __name__ == "__main__":
    main()
