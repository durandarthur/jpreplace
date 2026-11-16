"""Capture -> OCR -> group -> translate -> render."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PySide6.QtGui import QImage

from . import assets, config
from .layout import Block, group_lines, has_japanese
from .ocr import OCR
from .render import render
from .translate import LlamaServer, Translator


@dataclass
class Result:
    image: QImage
    blocks: list[Block]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())


class Pipeline:
    def __init__(self, settings: config.Settings) -> None:
        self.settings = settings
        self.ocr: OCR | None = None
        self.server: LlamaServer | None = None
        self.translator: Translator | None = None

    def load(self, status: Callable[[str], None] = lambda _msg: None) -> None:
        s = self.settings
        status("Checking models…")
        exe = assets.ensure_llama()
        det_dir, rec_dir = assets.ensure_ocr(s.ocr)
        model = assets.ensure_translator(s.translator)
        status("Loading OCR…")
        self.ocr = OCR(det_dir, rec_dir)
        status("Loading translator…")
        self.server = LlamaServer(exe, model, port=s.llama_port, parallel=s.parallel)
        self.server.start()
        self.translator = Translator(self.server.url, parallel=s.parallel)
        status("Warming up…")
        self.ocr(np.full((96, 320, 3), 255, dtype=np.uint8))
        self.translator.translate(["準備完了"], use_context=False)
        self.translator.cache.clear()

    def close(self) -> None:
        if self.translator:
            self.translator.close()
        if self.server:
            self.server.stop()

    def read(self, frame: np.ndarray) -> list[Block]:
        """OCR a BGR frame and return the Japanese text blocks."""
        assert self.ocr is not None
        return [b for b in group_lines(self.ocr(frame)) if has_japanese(b.text)]

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
