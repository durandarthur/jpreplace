"""Make the README pictures from the synthetic samples, with the real pipeline.

Usage: uv run python scripts/readme_images.py [--out docs/images]

The overlay hides itself from screen capture, so a screenshot of the running
app would not show the translation. Instead, the frame widget is drawn with
Qt and the pipeline's output is placed in its interior.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"  # draw the frame at 1:1, like the samples

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import make_samples  # noqa: E402

from jpreplace import app as overlay  # noqa: E402
from jpreplace.config import Settings  # noqa: E402
from jpreplace.pipeline import Pipeline  # noqa: E402


def to_array(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    rgb = np.frombuffer(image.constBits(), np.uint8, h * image.bytesPerLine()).reshape(h, image.bytesPerLine())
    return cv2.cvtColor(rgb[:, : w * 3].reshape(h, w, 3).copy(), cv2.COLOR_RGB2BGR)


def to_image(frame: np.ndarray) -> QImage:
    h, w = frame.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def translated(pipeline: Pipeline, frame: np.ndarray) -> tuple[QImage, int, float]:
    """The frame with the overlay drawn on it, the block count and the time."""
    pipeline.translator.cache.clear()
    pipeline.run(frame)  # warm-up for this size, as the app does after a resize
    pipeline.translator.cache.clear()
    start = time.perf_counter()
    result = pipeline.run(frame)
    elapsed = time.perf_counter() - start
    base = to_image(frame).convertToFormat(QImage.Format.Format_ARGB32)
    painter = QPainter(base)
    painter.drawImage(0, 0, result.image)
    painter.end()
    return base, len(result.blocks), elapsed


def frame_widget(width: int, height: int, status: str) -> QImage:
    settings = Settings()
    settings.save = lambda: None
    frame = overlay.Frame(settings)
    frame.resize(width, height)
    frame.bar.setGeometry(overlay.BORDER, overlay.BORDER, width - 2 * overlay.BORDER, overlay.BAR_HEIGHT)
    frame.bar.layout().activate()
    frame.translate_button.setEnabled(True)
    frame.live_button.setEnabled(True)
    frame.status.setText(status)
    image = frame.grab().toImage()
    frame.canvas.deleteLater()
    frame.deleteLater()
    return image


def hero(pipeline: Pipeline, out: Path) -> None:
    """The frame over a web page: English inside, the original outside."""
    page = to_array(make_samples.webpage())
    inner = QRect(40, 80, 870, 290)
    crop = page[inner.top() : inner.bottom() + 1, inner.left() : inner.right() + 1]
    english, blocks, seconds = translated(pipeline, crop)
    b, bar = overlay.BORDER, overlay.BAR_HEIGHT
    outer = QRect(inner.left() - b, inner.top() - b - bar, inner.width() + 2 * b, inner.height() + 2 * b + bar)
    chrome = frame_widget(outer.width(), outer.height(), overlay._summary(blocks, seconds))

    picture = to_image(page).convertToFormat(QImage.Format.Format_ARGB32)
    painter = QPainter(picture)
    painter.drawImage(outer.topLeft(), chrome)
    painter.drawImage(inner.topLeft(), english)
    painter.end()
    picture.copy(0, 0, picture.width(), 420).save(str(out / "frame.png"))


def before_after(pipeline: Pipeline, out: Path, name: str, source: QImage, crop: QRect | None = None) -> None:
    frame = to_array(source)
    english, _blocks, _seconds = translated(pipeline, frame)
    original = to_image(frame)
    if crop is not None:
        original, english = original.copy(crop), english.copy(crop)
    original.save(str(out / f"{name}-before.png"))
    english.save(str(out / f"{name}-after.png"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/images"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    pipeline = Pipeline(Settings.load())
    pipeline.load(print)
    try:
        hero(pipeline, args.out)
        before_after(pipeline, args.out, "game", make_samples.game(), QRect(0, 410, 1280, 278))
        before_after(pipeline, args.out, "manga", make_samples.manga(), QRect(30, 0, 840, 580))
        before_after(pipeline, args.out, "patterns", make_samples.hatched(), QRect(0, 0, 1000, 370))
    finally:
        pipeline.close()
    for path in sorted(args.out.glob("*.png")):
        print(path)
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
