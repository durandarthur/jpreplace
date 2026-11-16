"""Text detection + recognition with PP-OCRv6 ONNX models on ONNX Runtime."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
_REC_HEIGHT = 48
_REC_WIDTH_STEP = 96  # pad batch widths to multiples of this to limit distinct shapes
_REC_MAX_WIDTH = 3200


@dataclass
class Line:
    """One detected text line. `quad` is tl, tr, br, bl in image pixels."""

    quad: np.ndarray
    text: str
    score: float
    vertical: bool

    @property
    def box(self) -> tuple[float, float, float, float]:
        x0, y0 = self.quad.min(axis=0)
        x1, y1 = self.quad.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)


def _session(path: Path, use_gpu: bool) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers: list = ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        # HEURISTIC avoids a slow exhaustive cuDNN search on every new input shape.
        providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl."""
    pts = pts[np.argsort(pts[:, 0])]
    left, right = pts[:2], pts[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


_KANA = "ぁ-ゖァ-ヺ"
_KANA_DASH = re.compile(f"(?<=[{_KANA}])[-‐−–—](?=[{_KANA}]|$)")


def _fix_long_vowel(text: str) -> str:
    """A dash between kana is almost always a misread long-vowel mark."""
    return _KANA_DASH.sub("ー", text)


class OCR:
    def __init__(
        self,
        det_dir: Path,
        rec_dir: Path,
        use_gpu: bool = True,
        max_side: int = 2560,
        rec_batch: int = 32,
        min_score: float = 0.6,
        end_pad: float = 0.1,
        scale: float = 1.0,
    ) -> None:
        self.end_pad = end_pad
        self.scale = scale
        if use_gpu and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        det_cfg = yaml.safe_load((det_dir / "inference.yml").read_text(encoding="utf-8"))
        rec_cfg = yaml.safe_load((rec_dir / "inference.yml").read_text(encoding="utf-8"))
        post = det_cfg["PostProcess"]
        self.thresh = float(post["thresh"])
        self.box_thresh = float(post["box_thresh"])
        self.unclip_ratio = float(post["unclip_ratio"])
        self.max_side = max_side
        self.rec_batch = rec_batch
        self.min_score = min_score
        self.chars = ["", *[str(c) for c in rec_cfg["PostProcess"]["character_dict"]], " "]

        self.det = _session(det_dir / "inference.onnx", use_gpu)
        self.rec = _session(rec_dir / "inference.onnx", use_gpu)
        self.det_in = self.det.get_inputs()[0].name
        self.rec_in = self.rec.get_inputs()[0].name
        self.providers = self.det.get_providers()

    # -- detection ---------------------------------------------------------

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        h, w = bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        rh = max(32, int(round(h * scale / 32)) * 32)
        rw = max(32, int(round(w * scale / 32)) * 32)
        img = cv2.resize(bgr, (rw, rh)) if (rh, rw) != (h, w) else bgr
        x = (img.astype(np.float32) - _DET_MEAN) / _DET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        prob = self.det.run(None, {self.det_in: x})[0][0, 0]

        mask = (prob > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = w / rw, h / rh
        quads: list[np.ndarray] = []
        for contour in contours[:3000]:
            (cx, cy), (bw, bh), angle = cv2.minAreaRect(contour)
            if min(bw, bh) < 3:
                continue
            if self._box_score(prob, contour) < self.box_thresh:
                continue
            # Grow the shrunken DB kernel back to the full text extent.
            d = bw * bh * self.unclip_ratio / (2 * (bw + bh))
            # Extra margin along the reading direction: short labels otherwise
            # lose part of their first/last character.
            pad = self.end_pad * min(bw, bh)
            ew, eh = (pad, 0.0) if bw >= bh else (0.0, pad)
            pts = cv2.boxPoints(((cx, cy), (bw + 2 * d + 2 * ew, bh + 2 * d + 2 * eh), angle))
            if min(bw, bh) + 2 * d < 5:
                continue
            pts[:, 0] = np.clip(pts[:, 0] * sx, 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1] * sy, 0, h - 1)
            quads.append(_order_quad(pts))
        return quads

    @staticmethod
    def _box_score(prob: np.ndarray, contour: np.ndarray) -> float:
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour.reshape(-1, 2) - (x, y)], 1)
        return float(cv2.mean(prob[y : y + h, x : x + w], mask)[0])

    # -- recognition -------------------------------------------------------

    @staticmethod
    def _crop(bgr: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, bool]:
        tl, tr, br, bl = quad
        cw = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        ch = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
        cw, ch = max(cw, 1), max(ch, 1)
        dst = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(quad, dst)
        crop = cv2.warpPerspective(bgr, m, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        vertical = ch >= cw * 1.5
        if vertical:
            crop = np.rot90(crop)
        return crop, vertical

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        widths = [min(_REC_MAX_WIDTH, max(8, math.ceil(_REC_HEIGHT * c.shape[1] / c.shape[0]))) for c in crops]
        order = np.argsort(widths)
        for start in range(0, len(order), self.rec_batch):
            idx = order[start : start + self.rec_batch]
            bw = max(widths[i] for i in idx)
            bw = max(160, math.ceil(bw / _REC_WIDTH_STEP) * _REC_WIDTH_STEP)
            batch = np.zeros((len(idx), 3, _REC_HEIGHT, bw), dtype=np.float32)
            for row, i in enumerate(idx):
                img = cv2.resize(crops[i], (widths[i], _REC_HEIGHT), interpolation=cv2.INTER_LINEAR)
                batch[row, :, :, : widths[i]] = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
            probs = self.rec.run(None, {self.rec_in: batch})[0]
            ids = probs.argmax(axis=2)
            conf = probs.max(axis=2)
            for row, i in enumerate(idx):
                results[i] = self._ctc_decode(ids[row], conf[row])
        return results

    def _ctc_decode(self, ids: np.ndarray, conf: np.ndarray) -> tuple[str, float]:
        keep = ids != 0
        keep[1:] &= ids[1:] != ids[:-1]
        if not keep.any():
            return "", 0.0
        text = "".join(self.chars[i] for i in ids[keep])
        return text, float(conf[keep].mean())

    # -- full pipeline -----------------------------------------------------

    def __call__(self, bgr: np.ndarray) -> list[Line]:
        if self.scale != 1.0:  # upscaling helps with very small UI text
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        quads = self.detect(bgr)
        if not quads:
            return []
        cropped = [self._crop(bgr, q) for q in quads]
        texts = self.recognize([c for c, _ in cropped])
        lines = []
        for quad, (_, vertical), (text, score) in zip(quads, cropped, texts):
            text = _fix_long_vowel(text.strip())
            if text and score >= self.min_score:
                lines.append(Line(quad / self.scale, text, score, vertical))
        return lines
