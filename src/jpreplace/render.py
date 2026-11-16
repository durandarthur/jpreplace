"""Erase the original text and draw the translation in its place."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter

from .layout import Block

_MIN_FONT_PX = 9.0
_FONT_SCALE = 0.72  # English font size relative to the Japanese line box
_UNIFORM_TOL = 18
_WRAP = int(Qt.TextFlag.TextWordWrap)
_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR


@dataclass
class Patch:
    """What replaces one block: background pixels plus text styling."""

    x: int
    y: int
    pixels: np.ndarray  # BGR, covers the block's padded bounding box
    mask: np.ndarray  # uint8, 255 where `pixels` should be shown
    text_color: tuple[int, int, int]  # RGB
    fill: tuple[int, int, int]  # RGB, average background
    centred: bool
    # Free, evenly coloured space the translation may spill into (pixels).
    grow_left: int = 0
    grow_right: int = 0
    grow_down: int = 0


def _coons(top: np.ndarray, bottom: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Smoothly interpolate a patch from its four edges (h x w x 3)."""
    h, w = left.shape[0], top.shape[0]
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    ruled_v = (1 - v) * top[None] + v * bottom[None]
    ruled_u = (1 - u) * left[:, None] + u * right[:, None]
    corners = (
        (1 - u) * (1 - v) * top[0] + u * (1 - v) * top[-1] + (1 - u) * v * bottom[0] + u * v * bottom[-1]
    )
    return ruled_v + ruled_u - corners


def _smooth(edge: np.ndarray) -> np.ndarray:
    k = max(3, (len(edge) // 8) | 1)
    return cv2.blur(edge[None].astype(np.float32), (k, 1), borderType=cv2.BORDER_REPLICATE)[0]


def _free_run(strip: np.ndarray, colour: np.ndarray) -> int:
    """How many leading rows of `strip` (n x m x 3) match `colour` throughout."""
    if strip.size == 0:
        return 0
    ok = (np.abs(strip.astype(np.int16) - colour.astype(np.int16)).max(axis=2) < _UNIFORM_TOL).all(axis=1)
    bad = np.flatnonzero(~ok)
    return int(bad[0]) if len(bad) else len(ok)


def _text_colour(region: np.ndarray, dist: np.ndarray, background: np.ndarray) -> tuple[int, int, int]:
    """Median colour of the pixels that stand out most from the background."""
    peak = float(np.percentile(dist, 98)) if dist.size else 0.0
    if peak >= 90:
        b, g, r = np.median(region[dist >= 0.7 * peak], axis=0)
        return int(r), int(g), int(b)
    return (20, 20, 20) if float(background @ _LUMA) > 128 else (240, 240, 240)


def make_patch(frame: np.ndarray, block: Block) -> Patch:
    """Build the background patch that hides `block`'s original text."""
    fh, fw = frame.shape[:2]
    size = block.line_size
    pad = max(2, int(round(size * 0.1)))
    x0 = max(0, int(block.box[0]) - pad)
    y0 = max(0, int(block.box[1]) - pad)
    x1 = min(fw, int(np.ceil(block.box[2])) + pad)
    y1 = min(fh, int(np.ceil(block.box[3])) + pad)
    w, h = x1 - x0, y1 - y0
    region = frame[y0:y1, x0:x1]

    lines = np.zeros((h, w), dtype=np.uint8)
    for line in block.lines:
        lx0, ly0, lx1, ly1 = line.box
        cv2.rectangle(
            lines,
            (int(lx0) - x0 - pad, int(ly0) - y0 - pad),
            (int(np.ceil(lx1)) - x0 + pad, int(np.ceil(ly1)) - y0 + pad),
            255,
            -1,
        )
    inside = lines > 0

    centred = block.vertical
    if not block.vertical and len(block.lines) > 1:
        lefts = [ln.box[0] for ln in block.lines]
        centres = [(ln.box[0] + ln.box[2]) / 2 for ln in block.lines]
        centred = np.ptp(centres) < np.ptp(lefts)

    # Sample the background from a thin ring just outside the block.
    r = 2
    top = frame[max(0, y0 - r) : max(1, y0), x0:x1].astype(np.float32).mean(axis=0)
    bottom = frame[min(fh - 1, y1) : min(fh, y1 + r), x0:x1].astype(np.float32).mean(axis=0)
    left = frame[y0:y1, max(0, x0 - r) : max(1, x0)].astype(np.float32).mean(axis=1)
    right = frame[y0:y1, min(fw - 1, x1) : min(fw, x1 + r)].astype(np.float32).mean(axis=1)
    ring = np.concatenate([top, bottom, left, right])
    median = np.median(ring, axis=0)

    uniform = (np.abs(ring - median).max(axis=1) < _UNIFORM_TOL).mean() > 0.9
    if uniform:
        background = np.broadcast_to(median.astype(np.uint8), (h, w, 3))
        threshold = 30.0
    else:
        # Textured background: a median filter a little wider than the text
        # strokes removes them but keeps the scenery behind.
        k = int(min(255, max(5, size * 0.6))) | 1
        m = k
        ex0, ey0, ex1, ey1 = max(0, x0 - m), max(0, y0 - m), min(fw, x1 + m), min(fh, y1 + m)
        crop = (slice(y0 - ey0, y0 - ey0 + h), slice(x0 - ex0, x0 - ex0 + w))
        wide = frame[ey0:ey1, ex0:ex1]
        background = cv2.medianBlur(wide, k)[crop]
        threshold = 45.0

    dist = np.abs(region.astype(np.float32) - background).sum(axis=2)
    peak = float(np.percentile(dist[inside], 98)) if inside.any() else 0.0
    ink = ((dist > max(threshold, 0.25 * peak)) & inside).astype(np.uint8)
    # Marks that run off the edge of the block (rules, bubble outlines,
    # scenery) are not text: leave them alone.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, count):
        cx, cy, cw, ch, _area = stats[i]
        if cx == 0 or cy == 0 or cx + cw == w or cy + ch == h:
            ink[labels == i] = 0
    coverage = float(ink[inside].mean()) if inside.any() else 0.0
    mean_bg = background.reshape(-1, 3).mean(axis=0)
    fill = tuple(int(c) for c in mean_bg[::-1])
    colour = _text_colour(region[ink > 0], dist[ink > 0], mean_bg)

    if 0.01 < coverage < 0.6:
        grow = 5 + 2 * int(round(size / 32))
        mask = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))) * 255
        if not uniform:
            # Re-estimate the background with the strokes already removed, so
            # dense glyphs do not leave a ghost behind.
            cleaned = wide.copy()
            cleaned[crop][mask > 0] = background[mask > 0]
            background = cv2.medianBlur(cleaned, k)[crop]
        patch = Patch(x0, y0, background, mask, colour, fill, centred)
    else:
        # Could not separate text from background: replace the whole box.
        pixels = _coons(_smooth(top), _smooth(bottom), _smooth(left), _smooth(right))
        patch = Patch(x0, y0, np.clip(pixels, 0, 255).astype(np.uint8), np.full((h, w), 255, np.uint8),
                      colour, fill, centred)

    if uniform:
        # Note how far the same colour extends, so that longer English text
        # can use that room.
        margin = int(size * 0.4)
        reach_x = int(max(0.6 * w, 4 * size))
        reach_y = int(max(1.4 * size, 0.5 * h))
        run = _free_run(frame[y0:y1, x1 : x1 + reach_x].transpose(1, 0, 2), median)
        patch.grow_right = max(0, run - margin)
        run = _free_run(frame[y0:y1, max(0, x0 - reach_x) : x0][:, ::-1].transpose(1, 0, 2), median)
        patch.grow_left = max(0, run - margin)
        run = _free_run(frame[y1 : y1 + reach_y, x0:x1], median)
        patch.grow_down = max(0, run - margin)
    return patch


def _fit_font(font: QFont, text: str, w: float, h: float, start_px: float) -> tuple[float, bool]:
    """Largest pixel size <= start_px at which word-wrapped `text` fits w x h.
    Returns (size, fits)."""

    def fits(px: float) -> bool:
        font.setPixelSize(max(1, int(px)))
        rect = QFontMetricsF(font).boundingRect(QRectF(0, 0, w, 10_000), _WRAP, text)
        return rect.height() <= h + 1 and rect.width() <= w + 1

    hi = max(_MIN_FONT_PX, start_px)
    if fits(hi):
        return hi, True
    lo = _MIN_FONT_PX
    if not fits(lo):
        return lo, False
    while hi - lo > 0.75:
        mid = (lo + hi) / 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo, True


def _harmonise(blocks: list[Block]) -> list[float]:
    """Text size per block, evened out between neighbours: blocks of nearly
    equal size sharing a row or a column (menu items, buttons, paragraphs)
    get the same size instead of slightly different fonts."""

    def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
        return min(a1, b1) - max(a0, b0) >= 0.5 * min(a1 - a0, b1 - b0)

    sizes = [b.line_size for b in blocks]
    out = []
    for a, size in zip(blocks, sizes):
        peers = sorted(
            s
            for b, s in zip(blocks, sizes)
            if abs(s - size) <= 0.2 * max(s, size)
            and (overlaps(a.box[1], a.box[3], b.box[1], b.box[3]) or overlaps(a.box[0], a.box[2], b.box[0], b.box[2]))
        )
        out.append(peers[len(peers) // 2])
    return out


def render(frame: np.ndarray, blocks: list[Block], font_family: str = "Segoe UI") -> QImage:
    """Return a transparent overlay image the size of `frame` with every
    translated block erased and re-drawn in English."""
    fh, fw = frame.shape[:2]
    canvas = np.zeros((fh, fw, 4), dtype=np.uint8)
    blocks = [b for b in blocks if b.translation]
    patches = [make_patch(frame, b) for b in blocks]
    for patch in patches:
        h, w = patch.mask.shape
        target = canvas[patch.y : patch.y + h, patch.x : patch.x + w]
        visible = patch.mask > 0
        target[..., :3][visible] = patch.pixels[visible]
        target[..., 3][visible] = 255

    image = QImage(canvas.data, fw, fh, fw * 4, QImage.Format.Format_ARGB32).copy()
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    font = QFont(font_family)
    sizes = _harmonise(blocks)
    for block, patch, size in zip(blocks, patches, sizes):
        text = block.translation
        target_px = size * _FONT_SCALE
        inset = 0.0 if patch.centred else min(size * 0.25, (block.box[2] - block.box[0]) * 0.1)
        base = QRectF(block.box[0] + inset, block.box[1], block.box[2] - block.box[0] - inset,
                      block.box[3] - block.box[1])
        side = min(patch.grow_left, patch.grow_right) if patch.centred else 0
        candidates = [
            base,
            base.adjusted(-side, 0, side, 0) if patch.centred else base.adjusted(0, 0, patch.grow_right, 0),
            base.adjusted(0, 0, 0, patch.grow_down),
        ]
        best, best_px, best_fits = base, 0.0, False
        for rect in candidates:
            px, fits = _fit_font(font, text, rect.width(), rect.height(), target_px)
            if (fits, px) > (best_fits, best_px):
                best, best_px, best_fits = rect, px, fits
            if fits and px >= 0.9 * target_px:
                break
        font.setPixelSize(max(1, int(best_px)))
        painter.setFont(font)

        if not best_fits:
            # Too long even at the minimum size: spill onto a backing plate.
            wide = max(best.width(), min(fw - best.x(), best.width() * 1.6))
            need = QFontMetricsF(font).boundingRect(QRectF(0, 0, wide, 10_000), _WRAP, text)
            best = QRectF(best.x(), best.y(), max(best.width(), need.width()), max(best.height(), need.height()))
            painter.fillRect(best.adjusted(-3, -1, 3, 1), QColor(*patch.fill))

        horizontal = Qt.AlignmentFlag.AlignHCenter if patch.centred else Qt.AlignmentFlag.AlignLeft
        multi = len(block.lines) > 1 or best.height() > base.height()
        vertical = Qt.AlignmentFlag.AlignTop if multi and not patch.centred else Qt.AlignmentFlag.AlignVCenter
        painter.setPen(QColor(*patch.text_color))
        painter.drawText(best, _WRAP | int(horizontal | vertical), text)
    painter.end()
    return image
