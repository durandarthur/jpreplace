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
_REC_WIDTH_STEP = 128  # batch widths are padded to multiples of this
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


def _argmax_model(path: Path) -> Path:
    """A copy of the recogniser that returns the best class and its score per
    time step instead of the full probability table. The table is hundreds of
    megabytes per batch, and copying it off the GPU dominated OCR time."""
    out = path.with_name("inference.argmax.onnx")
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(path))
    probs = model.graph.output[0].name
    model.graph.node.extend(
        [
            helper.make_node("ArgMax", [probs], ["ids"], axis=2, keepdims=0),
            helper.make_node("ReduceMax", [probs], ["conf"], axes=[2], keepdims=0),
        ]
    )
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("ids", TensorProto.INT64, None),
            helper.make_tensor_value_info("conf", TensorProto.FLOAT, None),
        ]
    )
    onnx.save(model, str(out))
    return out


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl."""
    pts = pts[np.argsort(pts[:, 0])]
    left, right = pts[:2], pts[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _snap(quad: np.ndarray) -> np.ndarray:
    """Replace a (nearly) axis-aligned quad by its integer bounding box."""
    x0, y0 = np.floor(quad.min(axis=0))
    x1, y1 = np.ceil(quad.max(axis=0))
    if (x1 - x0) * (y1 - y0) > 1.1 * cv2.contourArea(quad) + 16:
        return quad
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _is_upright(quad: np.ndarray) -> bool:
    tl, tr, br, bl = quad
    return tl[1] == tr[1] and bl[1] == br[1] and tl[0] == bl[0] and tr[0] == br[0]


_KANA ="ぁ-ゖァ-ヺ"
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
        rec_batch: int = 8,  # small batches of similar width waste less padding
        min_score: float = 0.6,
        end_pad: float = 0.3,
        side_pad: float = 0.1,
        scale: float = 1.0,
    ) -> None:
        self.end_pad = end_pad
        self.side_pad = side_pad
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
        self.rec = _session(_argmax_model(rec_dir / "inference.onnx"), use_gpu)
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
            thickness = min(bw, bh) + 2 * d
            if thickness < 5:
                continue
            # The recogniser wants a little air around the glyphs, and more of
            # it along the reading direction, where tight boxes clip characters.
            end, side = self.end_pad * thickness, self.side_pad * thickness
            ew, eh = (end, side) if bw >= bh else (side, end)
            pts = cv2.boxPoints(((cx, cy), (bw + 2 * d + 2 * ew, bh + 2 * d + 2 * eh), angle))
            pts[:, 0] = np.clip(pts[:, 0] * sx, 0, w)
            pts[:, 1] = np.clip(pts[:, 1] * sy, 0, h)
            quads.append(_snap(_order_quad(pts)))
        return quads

    def _extend_ends(self, bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Grow an upright box over trailing punctuation the detector missed
        (typically the 。 that ends a vertical column)."""
        if not _is_upright(quad):
            return quad
        h, w = bgr.shape[:2]
        x0, y0 = quad[0].astype(int)
        x1, y1 = quad[2].astype(int)
        vertical = (y1 - y0) >= 1.5 * (x1 - x0)
        thickness = (x1 - x0) if vertical else (y1 - y0)
        reach = int(thickness * 0.9)
        if vertical:
            ext = bgr[y1 : min(h, y1 + reach), x0:x1]
        else:
            ext = bgr[y0:y1, x1 : min(w, x1 + reach)].transpose(1, 0, 2)
        if ext.shape[0] < 4 or ext.shape[1] < 4:
            return quad
        background = np.median(bgr[y0:y1, x0:x1].reshape(-1, 3), axis=0)
        ink = (np.abs(ext.astype(np.int16) - background.astype(np.int16)).max(axis=2) > 60).sum(axis=1)
        rows = np.flatnonzero(ink)
        if len(rows) == 0 or rows[0] > 0.6 * reach:
            return quad
        # Take the first run of inked rows; it only counts as punctuation if it
        # is small and followed by clear space.
        breaks = np.flatnonzero(np.diff(rows) > 3)
        last = int(rows[breaks[0]] if len(breaks) else rows[-1])
        if last >= ext.shape[0] - 3 or last - rows[0] > 0.7 * thickness or ink[: last + 1].sum() < 4:
            return quad
        grow = last + 3
        quad = quad.copy()
        if vertical:
            quad[2:, 1] = min(h, y1 + grow)
        else:
            quad[1:3, 0] = min(w, x1 + grow)
        return quad

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
        if _is_upright(quad):
            # Plain slicing keeps small screen text pixel-exact.
            crop = bgr[int(tl[1]) : int(br[1]), int(tl[0]) : int(br[0])]
            ch, cw = crop.shape[:2]
        else:
            cw = max(1, int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))))
            ch = max(1, int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))))
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
            bw = math.ceil(max(widths[i] for i in idx) / _REC_WIDTH_STEP) * _REC_WIDTH_STEP
            # Always a full batch: together with the width buckets this keeps
            # the set of input shapes small, and each new shape costs a pause.
            batch = np.zeros((self.rec_batch, 3, _REC_HEIGHT, bw), dtype=np.float32)
            for row, i in enumerate(idx):
                img = cv2.resize(crops[i], (widths[i], _REC_HEIGHT), interpolation=cv2.INTER_LINEAR)
                batch[row, :, :, : widths[i]] = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
            ids, conf = self.rec.run(None, {self.rec_in: batch})
            for row, i in enumerate(idx):
                results[i] = self._ctc_decode(ids[row], conf[row])
        return results

    def warm_up(self, max_width: int = 2560) -> None:
        """Run every common recogniser input shape once. The GPU backend tunes
        itself per shape on first use, which would otherwise delay the first
        translations by up to a second."""
        for width in range(_REC_WIDTH_STEP, max_width + 1, _REC_WIDTH_STEP):
            self.rec.run(None, {self.rec_in: np.zeros((self.rec_batch, 3, _REC_HEIGHT, width), np.float32)})

    def warm_up_detector(self, height: int, width: int) -> None:
        """Same, for the detector at a given capture size."""
        self.detect(np.zeros((max(32, height), max(32, width), 3), np.uint8))

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
        quads = [self._extend_ends(bgr, q) for q in self.detect(bgr)]
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
