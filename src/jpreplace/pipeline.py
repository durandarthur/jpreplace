"""Capture -> OCR -> group -> translate -> render."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PySide6.QtGui import QImage

from . import assets, config
from .layout import Block, group_lines, has_japanese
from .ocr import OCR, Line
from .render import render
from .translate import Backend, LlamaServer, Translator


@dataclass
class Result:
    image: QImage
    blocks: list[Block]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())


def same_layout(a: list[Block], b: list[Block], tolerance: float = 6.0) -> bool:
    """True if two OCR passes found the same text in the same places."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.text != y.text or max(abs(p - q) for p, q in zip(x.box, y.box)) > tolerance:
            return False
    return True


class Pipeline:
    def __init__(self, settings: config.Settings) -> None:
        self.settings = settings
        self.ocr: OCR | None = None
        self.servers: list[LlamaServer] = []
        self.translator: Translator | None = None

    def load(self, status: Callable[[str], None] = lambda _msg: None) -> None:
        s = self.settings
        status("Checking models…")
        exe = assets.ensure_llama(status)
        det_dir, rec_dir = assets.ensure_ocr(s.ocr, status)
        # "auto" keeps both models loaded: the 8B for quality, and the 2B to
        # stay fast when a screen holds more text than the 8B can do in time.
        names = ["plamo-8b", "plamo-2b"] if s.translator == "auto" else [s.translator]
        models = [assets.ensure_translator(name, status) for name in names]
        status("Loading OCR…")
        self.ocr = OCR(det_dir, rec_dir, scale=s.ocr_scale)
        status("Loading translator…")
        backends = []
        for i, model in enumerate(models):
            server = LlamaServer(exe, model, port=s.llama_port + i, parallel=s.parallel)
            self.servers.append(server)
            server.start()
            limit = s.quality_max_chars if i < len(models) - 1 else 0
            backends.append(Backend(server.url, s.parallel, limit))
        self.translator = Translator(backends)
        status("Warming up…")
        self.ocr.warm_up()
        self.translator.warm_up()
        # First use of the text renderer loads fonts; do it now, not on the
        # first translation.
        sample = np.full((96, 320, 3), 255, dtype=np.uint8)
        quad = np.array([[8, 8], [200, 8], [200, 40], [8, 40]], dtype=np.float32)
        block = group_lines([Line(quad, "準備", 1.0, False)])[0]
        block.translation = "Ready"
        render(sample, [block], s.font_family)

    def prepare(self, height: int, width: int) -> None:
        """Get the detector ready for captures of this size."""
        assert self.ocr is not None
        scale = self.settings.ocr_scale
        self.ocr.warm_up_detector(int(height * scale), int(width * scale))

    def close(self) -> None:
        if self.translator:
            self.translator.close()
        for server in self.servers:
            server.stop()

    def read(self, frame: np.ndarray) -> list[Block]:
        """OCR a BGR frame and return the Japanese text blocks."""
        assert self.ocr is not None
        return [b for b in group_lines(self.ocr(frame)) if has_japanese(b.text)]

    def fill_cached(self, blocks: list[Block]) -> int:
        """Apply already-known translations; return how many blocks lack one."""
        assert self.translator is not None
        missing = 0
        for block in blocks:
            hit = self.translator.cached(block.text)
            if hit is None:
                missing += 1
            else:
                block.translation = hit
        return missing

    def draw(self, frame: np.ndarray, blocks: list[Block]) -> QImage:
        return render(frame, blocks, self.settings.font_family)

    def run(self, frame: np.ndarray, blocks: list[Block] | None = None) -> Result:
        assert self.translator is not None
        timings: dict[str, float] = {}
        t = time.perf_counter()
        if blocks is None:
            blocks = self.read(frame)
            timings["ocr"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        translations = self.translator.translate([b.text for b in blocks])
        for block, translation in zip(blocks, translations):
            block.translation = translation
        timings["translate"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        image = render(frame, blocks, self.settings.font_family)
        timings["render"] = (time.perf_counter() - t) * 1000
        return Result(image, blocks, timings)
