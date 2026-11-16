"""Run the full pipeline on image files and report timings.

Usage: uv run python scripts/translate_image.py IMAGE [IMAGE...] [--out DIR] [--repeat N]

Writes `<name>.en.png` (the image with Japanese replaced by English).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QGuiApplication, QImage, QPainter

from jpreplace.config import Settings
from jpreplace.pipeline import Pipeline


def composite(frame: np.ndarray, overlay: QImage) -> QImage:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).convertToFormat(QImage.Format.Format_ARGB32)
    painter = QPainter(base)
    painter.drawImage(0, 0, overlay)
    painter.end()
    return base


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--repeat", type=int, default=1, help="extra uncached runs for timing")
    args = p.parse_args()

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    pipeline = Pipeline(Settings.load())
    pipeline.load(print)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        for path in args.images:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"cannot read {path}")
                continue
            for _ in range(max(1, args.repeat)):
                pipeline.translator.cache.clear()
                result = pipeline.run(frame)
                stages = ", ".join(f"{k} {v:.0f}" for k, v in result.timings.items())
                print(f"{path.name}: {result.total_ms:.0f} ms ({stages}) for {len(result.blocks)} blocks")
            for block in result.blocks:
                print(f"   {block.text}\n     -> {block.translation}")
            composite(frame, result.image).save(str(args.out / f"{path.stem}.en.png"))
    finally:
        pipeline.close()
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
