"""Capture -> OCR -> group -> translate -> render."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
from PySide6.QtGui import QImage

from . import assets, config
from .layout import Block, group_lines, has_japanese
from .ocr import OCR, Line
from .render import render
from .translate import Backend, LlamaServer, Translator, api_base, list_models

_SYMBOLS = re.compile(r"[\W_]+")


@dataclass
class Result:
    image: QImage
    blocks: list[Block]
    timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None  # why some text could not be translated

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())


def _words(text: str) -> str:
    """Text without punctuation, symbols and spaces: OCR flickers on those
    (a blinking ▼, an ellipsis read as dots) far more than on words."""
    return _SYMBOLS.sub("", text)


def same_layout(a: list[Block], b: list[Block], tolerance: float = 6.0) -> bool:
    """True if two OCR passes found the same text in the same places."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if _words(x.text) != _words(y.text) or max(abs(p - q) for p, q in zip(x.box, y.box)) > tolerance:
            return False
    return True


class Pipeline:
    def __init__(self, settings: config.Settings) -> None:
        self.settings = settings
        self.ocr: OCR | None = None
        self.servers: list[LlamaServer] = []
        self.translator: Translator | None = None
        self.translator_name = ""

    def load(self, status: Callable[[str], None] = lambda _msg: None) -> None:
        s = self.settings
        if self.ocr is None:
            status("Checking models…")
            det_dir, rec_dir = assets.ensure_ocr(s.ocr, status)
            status("Loading OCR…")
            ocr = OCR(det_dir, rec_dir, scale=s.ocr_scale)
            status("Warming up…")
            ocr.warm_up()
            # First use of the text renderer loads fonts; do it now, not on
            # the first translation.
            sample = np.full((96, 320, 3), 255, dtype=np.uint8)
            quad = np.array([[8, 8], [200, 8], [200, 40], [8, 40]], dtype=np.float32)
            block = group_lines([Line(quad, "準備", 1.0, False)])[0]
            block.translation = "Ready"
            render(sample, [block], s.font_family)
            self.ocr = ocr
        self.load_translator(status)

    def load_translator(self, status: Callable[[str], None] = lambda _msg: None) -> None:
        """(Re)start translation as the settings say, replacing what runs now."""
        self.close_translator()
        s = self.settings
        if s.translator == "server":
            backends = [self._server_backend(status)]
            self.translator_name = backends[0].model or backends[0].url
        else:
            exe = assets.ensure_llama(status)
            if s.translator == "file":
                model = Path(s.model_file.strip().strip('"')).expanduser()
                if not model.is_file():
                    raise RuntimeError(f"model file not found: {s.model_file or '(none chosen)'}")
                models = [model]
            else:
                # "auto" keeps both PLaMo models loaded: the 8B for quality,
                # and the 2B to stay fast when a screen holds more text than
                # the 8B can do in time.
                names = s.builtin_translators()
                if not names:
                    raise RuntimeError(f"unknown translator {s.translator!r}")
                models = [assets.ensure_translator(name, status) for name in names]
            status("Loading translator…")
            backends = []
            for i, model in enumerate(models):
                style = s.resolved_prompt_style(model.name)
                server = LlamaServer(
                    exe, model, port=s.llama_port + i, parallel=s.parallel,
                    gpu_layers=s.gpu_layers, chat=style == "chat",
                )
                self.servers.append(server)
                server.start()
                limit = s.quality_max_chars if i < len(models) - 1 else 0
                backends.append(Backend(server.url, s.parallel, limit, style=style, system_prompt=s.system_prompt))
            self.translator_name = " + ".join(m.stem for m in models)
        translator = Translator(backends)
        status("Warming up translator…")
        try:
            translator.warm_up()
        except Exception:
            translator.close()
            self.close_translator()
            raise
        self.translator = translator

    def _server_backend(self, status: Callable[[str], None]) -> Backend:
        s = self.settings
        if not s.server_url.strip():
            raise RuntimeError("no server URL set")
        url = api_base(s.server_url)
        status(f"Connecting to {url}…")
        model = s.server_model.strip()
        if not model:
            try:
                available = list_models(url, s.server_api_key)
            except Exception as exc:  # some servers have no model list; the warm-up will tell
                available = []
                if isinstance(exc, httpx.ConnectError):
                    raise RuntimeError(f"cannot connect to {url}") from exc
            model = available[0] if available else ""
        return Backend(
            url, s.parallel, style=s.resolved_prompt_style(model), native=False, model=model,
            api_key=s.server_api_key, system_prompt=s.system_prompt, extra=dict(s.server_extra),
        )

    def close_translator(self) -> None:
        if self.translator:
            self.translator.close()
            self.translator = None
        for server in self.servers:
            server.stop()
        self.servers.clear()

    def prepare(self, height: int, width: int) -> None:
        """Get the detector ready for captures of this size."""
        assert self.ocr is not None
        scale = self.settings.ocr_scale
        self.ocr.warm_up_detector(int(height * scale), int(width * scale))

    def close(self) -> None:
        self.close_translator()

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
        return Result(image, blocks, timings, self.translator.last_error)
