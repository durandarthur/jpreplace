"""Compare translation speed across models and parallelism on real screenshots.

Usage: uv run python scripts/bench_translate.py IMAGE [IMAGE...]
           [--models plamo-8b plamo-2b] [--parallel 8 16] [--show]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from jpreplace import assets, config
from jpreplace.layout import group_lines, has_japanese
from jpreplace.ocr import OCR
from jpreplace.translate import Backend, LlamaServer, Translator


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--models", nargs="+", default=list(config.TRANSLATORS))
    p.add_argument("--parallel", nargs="+", type=int, default=[16])
    p.add_argument("--show", action="store_true", help="print the translations")
    args = p.parse_args()

    det, rec = assets.ensure_ocr("medium")
    ocr = OCR(det, rec)
    pages: dict[str, list[str]] = {}
    for path in args.images:
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        texts = [b.text for b in group_lines(ocr(frame)) if has_japanese(b.text)]
        pages[path.name] = texts
        print(f"{path.name}: {len(texts)} blocks, {sum(map(len, texts))} characters")
    del ocr

    for model in args.models:
        for parallel in args.parallel:
            server = LlamaServer(assets.ensure_llama(), assets.ensure_translator(model), port=8770, parallel=parallel)
            server.start()
            translator = Translator([Backend(server.url, parallel)])
            try:
                translator.translate(["準備完了"], use_context=False)
                for name, texts in pages.items():
                    for use_context in (True, False):
                        translator.cache.clear()
                        start = time.perf_counter()
                        out = translator.translate(texts, use_context=use_context)
                        seconds = time.perf_counter() - start
                        words = sum(len(t.split()) for t in out)
                        print(
                            f"{model:9} np={parallel:<3} context={use_context!s:5} {name:14} "
                            f"{seconds:6.2f} s  ({words} words)"
                        )
                    if args.show:
                        for source, target in zip(texts, out):
                            print(f"    {source}\n      -> {target}")
            finally:
                translator.close()
                server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
