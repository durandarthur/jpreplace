"""Erase the original text and draw the translation in its place."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter

from .layout import Block

_MIN_FONT_PX = 9.0
_FONT_SCALE = 0.72  # English font size relative to the Japanese line box
_UNIFORM_TOL = 18
_WRAP = int(Qt.TextFlag.TextWordWrap)
_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR
# Repeating backgrounds (see _pattern_fill).
_PERIOD_MAX = 32  # longest pattern repeat looked for, in pixels
_PATTERN_MIN_VAR = 60.0  # flatter surroundings are handled as plain or textured
_PATTERN_MAX_ERROR = 0.12  # error allowed for a shift, relative to the background's variance
_MAX_SOURCES = 150
_ESTIMATE_SPAN = 320  # length of block the pattern is measured along
_NEAR = 16  # width of the background ring around the text that shifts are judged on


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
    # RGB halo drawn around the text, for busy backgrounds (screentone, hatching).
    outline: tuple[int, int, int] | None = None


Rect = tuple[int, int, int, int]  # x0, y0, x1, y1


def _shift_errors(
    img: np.ndarray, ring: np.ndarray, clean: np.ndarray, ry: int, rx: int
) -> tuple[np.ndarray, np.ndarray]:
    """For every shift (dx, dy) with |dy| <= ry and |dx| <= rx: the summed
    squared difference between the `ring` pixels of `img` and the `clean`
    pixels that far away, and how many such pixel pairs there are. A masked
    correlation, computed with FFTs. Arrays are indexed [dy + ry, dx + rx]."""
    # Enough padding that no shift wraps around, rounded up to sizes the
    # FFT is fast for.
    shape = tuple(cv2.getOptimalDFTSize(n + r) for n, r in zip(img.shape, (ry, rx)))
    a, b = ring.astype(np.float64), clean.astype(np.float64)
    fa, fai, fai2, fb, fbi, fbi2 = (
        np.fft.rfft2(x, shape) for x in (a, a * img, a * img * img, b, b * img, b * img * img)
    )
    # Correlations sum_p a(p) b(p + d) are conj(A) * B in the frequency domain.
    pairs = np.fft.irfft2(np.conj(fa) * fb, shape)
    ssd = np.fft.irfft2(np.conj(fai2) * fb + np.conj(fa) * fbi2 - 2 * np.conj(fai) * fbi, shape)
    index = np.ix_(np.arange(-ry, ry + 1) % shape[0], np.arange(-rx, rx + 1) % shape[1])
    return ssd[index], pairs[index]


def _pattern_fill(
    frame: np.ndarray, gray: np.ndarray, rect: Rect, avoid: list[Rect]
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """Rebuild the background under `rect` if it is a repeating pattern
    (hatching, screentone, stripes, a tiled texture).

    Every shift up to one block-height above/below (wide text) or one
    block-width aside (tall text) is scored by how well it maps the text-free
    pixels around `rect` onto each other. Each pixel under the text is then
    copied from the best-scoring shift that lands on text-free background,
    so the pattern continues in phase even between tightly packed lines.

    Returns the pixels and the pattern's shortest repeats along x and y
    (for `_pattern_reach`), or None if the background is not a pattern."""
    fh, fw = gray.shape
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    ry, rx = (h + _PERIOD_MAX, _PERIOD_MAX) if w >= h else (_PERIOD_MAX, w + _PERIOD_MAX)
    wx0, wy0, wx1, wy1 = max(0, x0 - rx), max(0, y0 - ry), min(fw, x1 + rx), min(fh, y1 + ry)
    img = gray[wy0:wy1, wx0:wx1].astype(np.float64)
    clean = np.ones(img.shape, dtype=bool)
    for ax0, ay0, ax1, ay1 in [rect, *avoid]:
        clean[max(0, ay0 - wy0) : max(0, ay1 - wy0), max(0, ax0 - wx0) : max(0, ax1 - wx0)] = False
    # Shifts are judged on the background right around the text, so that
    # a pattern is only continued from where it is the same pattern (not
    # from beyond the edge of a dialogue box, say).
    ring = np.zeros(img.shape, dtype=bool)
    ring[
        max(0, y0 - wy0 - _NEAR) : y1 - wy0 + _NEAR,
        max(0, x0 - wx0 - _NEAR) : x1 - wx0 + _NEAR,
    ] = True
    ring &= clean
    if int(ring.sum()) < 400:
        return None
    variance = float(img[ring].var())
    if variance < _PATTERN_MIN_VAR:
        return None

    # The shift statistics need only part of a long block: its two ends,
    # whose margins are often the only background beside every line.
    long_axis, reach = (1, rx) if w >= h else (0, ry)
    length = img.shape[long_axis]
    end = _ESTIMATE_SPAN // 2 + 2 * reach
    spans = [slice(0, end), slice(length - end, length)] if length > 2 * end else [slice(None)]
    ssd = pairs = 0.0
    count = 0
    for span in spans:
        part = (slice(None), span) if long_axis == 1 else (span, slice(None))
        part_ssd, part_pairs = _shift_errors(img[part], ring[part], clean[part], ry, rx)
        ssd, pairs, count = ssd + part_ssd, pairs + part_pairs, count + int(ring[part].sum())
    errors = ssd / np.maximum(pairs, 1.0)
    dy, dx = np.mgrid[-ry : ry + 1, -rx : rx + 1]
    good = (pairs >= max(200, 0.2 * count)) & (errors <= _PATTERN_MAX_ERROR * variance) & ((dx != 0) | (dy != 0))
    if not good.any():
        return None
    # Best match first; among equals, the nearest source.
    score = np.where(good, errors / variance + 0.0005 * np.hypot(dx, dy), np.inf)
    # Only moving across the short side gets a source off the text, so keep
    # the best shift per offset in that direction (a pattern has many
    # equally good ones, nearly all of them right next to each other).
    across = 1 if w >= h else 0
    pick = np.argmin(score, axis=across)
    lines = np.arange(score.shape[1 - across])
    index = (lines, pick) if across == 1 else (pick, lines)
    keep = np.isfinite(score[index])
    order = np.argsort(score[index][keep])[:_MAX_SOURCES]
    sources = list(zip(dx[index][keep][order].tolist(), dy[index][keep][order].tolist()))

    # Fill, keeping track of just the pixels still to do.
    rows, cols = (a.ravel() for a in np.mgrid[y0 - wy0 : y1 - wy0, x0 - wx0 : x1 - wx0])
    targets = np.arange(h * w)
    pixels = np.zeros((h * w, 3), dtype=np.uint8)
    window_h, window_w = img.shape
    for sx, sy in sources:
        r, c = rows + sy, cols + sx
        ok = (r >= 0) & (r < window_h) & (c >= 0) & (c < window_w)
        ok[ok] = clean[r[ok], c[ok]]
        if ok.any():
            pixels[targets[ok]] = frame[wy0 + r[ok], wx0 + c[ok]]
            rows, cols, targets = rows[~ok], cols[~ok], targets[~ok]
            if not len(targets):
                break
    if len(targets):
        return None
    pixels = pixels.reshape(h, w, 3)

    repeats = []
    for along in ((dy == 0) & good, (dx == 0) & good):
        if along.any():
            i = np.unravel_index(int(np.argmin(np.where(along, score, np.inf))), score.shape)
            repeats.append((int(dx[i]), int(dy[i])))
    return pixels, repeats


def _pattern_reach(
    frame: np.ndarray, rect: Rect, shifts: list[tuple[int, int]], reach_x: int, reach_y: int
) -> tuple[int, int, int]:
    """How far the pattern continues unbroken (no text, no panel border)
    left of, right of and below `rect`: where the frame still matches itself
    under every one of `shifts`. (One shift alone is not enough: a border
    parallel to it would match itself.)"""
    if not shifts:
        return 0, 0, 0
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = rect
    wx0 = max([0, x0 - reach_x] + [-dx for dx, _ in shifts])
    wx1 = min([fw, x1 + reach_x] + [fw - dx for dx, _ in shifts])
    wy0 = max([0, y0] + [-dy for _, dy in shifts])
    wy1 = min([fh, y1 + reach_y] + [fh - dy for _, dy in shifts])
    if wx0 > x0 or wy0 > y0 or wx1 < x1 or wy1 < y1:
        return 0, 0, 0
    a = frame[wy0:wy1, wx0:wx1].astype(np.int16)
    same = np.ones(a.shape[:2], dtype=bool)
    for dx, dy in shifts:
        b = frame[wy0 + dy : wy1 + dy, wx0 + dx : wx1 + dx].astype(np.int16)
        same &= np.abs(a - b).max(axis=2) < 40

    def run(ok: np.ndarray) -> int:
        bad = np.flatnonzero(~ok)
        return int(bad[0]) if len(bad) else len(ok)

    rows = slice(y0 - wy0, y1 - wy0)
    right = run(same[rows, x1 - wx0 :].mean(axis=0) > 0.85)
    left = run(same[rows, : x0 - wx0][:, ::-1].mean(axis=0) > 0.85)
    down = run(same[y1 - wy0 :, x0 - wx0 : x1 - wx0].mean(axis=1) > 0.85)
    return left, right, down


def _busy(pixels: np.ndarray) -> bool:
    """Dense fine detail (hatching, screentone), as opposed to a few broad
    areas of colour: a sizeable share of the pixels sit on an edge."""
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return float((edges > 120).mean()) > 0.1


def _outlined_colours(
    region: np.ndarray, ink: np.ndarray
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """(text, outline) RGB colours if the ink is outlined lettering (dark
    glyphs ringed with light, or the reverse), else None. The glyph colour
    is the one buried deeper inside the ink."""
    pixels = region[ink > 0].astype(np.float32)
    if len(pixels) < 50:
        return None
    luma = pixels @ _LUMA
    dark, light = luma < 90, luma > 170
    if dark.mean() < 0.15 or light.mean() < 0.15:
        return None
    depth = cv2.distanceTransform(ink, cv2.DIST_L2, 3)[ink > 0]
    inner, outer = (dark, light) if depth[dark].mean() > depth[light].mean() else (light, dark)
    b, g, r = np.median(pixels[inner], axis=0)
    ob, og, orr = np.median(pixels[outer], axis=0)
    return (int(r), int(g), int(b)), (int(orr), int(og), int(ob))


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
        # The dominant ink colour first (so a few darker marks next to the
        # text, like a bullet or an arrow, do not win), then the solid core
        # of that colour rather than its anti-aliased fringe.
        pixels = region.astype(np.float32)
        dominant = np.median(pixels[dist >= 0.5 * peak], axis=0)
        family = np.abs(pixels - dominant).sum(axis=1) < 120
        if family.any():
            core = dist[family] >= 0.8 * np.percentile(dist[family], 95)
            b, g, r = np.median(pixels[family][core], axis=0)
        else:
            b, g, r = dominant
        return int(r), int(g), int(b)
    return (20, 20, 20) if float(background @ _LUMA) > 128 else (240, 240, 240)


def _padded_rect(block: Block, fw: int, fh: int) -> Rect:
    pad = max(2, int(round(block.line_size * 0.1)))
    return (
        max(0, int(block.box[0]) - pad),
        max(0, int(block.box[1]) - pad),
        min(fw, int(np.ceil(block.box[2])) + pad),
        min(fh, int(np.ceil(block.box[3])) + pad),
    )


def _pattern_patch(
    frame: np.ndarray, gray: np.ndarray, block: Block, avoid: list[Rect], centred: bool
) -> Patch | None:
    """Rebuild the background of `block` from elsewhere in a repeating
    pattern; None if the background is not one."""
    fh, fw = frame.shape[:2]
    # A little extra margin: lettering on patterns is often outlined, and
    # the outline reaches beyond the detector's box.
    grow = max(2, int(round(block.line_size * 0.12)))

    def widen(r: Rect) -> Rect:
        return max(0, r[0] - grow), max(0, r[1] - grow), min(fw, r[2] + grow), min(fh, r[3] + grow)

    rect = widen(_padded_rect(block, fw, fh))
    found = _pattern_fill(frame, gray, rect, [widen(r) for r in avoid])
    if found is None:
        return None
    pixels, repeats = found
    x0, y0, x1, y1 = rect
    region = frame[y0:y1, x0:x1]
    ink = (np.abs(region.astype(np.int16) - pixels.astype(np.int16)).sum(axis=2) > 90).astype(np.uint8)
    mean_bg = pixels.reshape(-1, 3).astype(np.float32).mean(axis=0)
    fill = tuple(int(c) for c in mean_bg[::-1])
    outlined = _outlined_colours(region, ink)
    if outlined:
        colour, outline = outlined
    else:
        dist = np.abs(region.astype(np.float32) - pixels).sum(axis=2)
        colour = _text_colour(region[ink > 0], dist[ink > 0], mean_bg)
        outline = None
        if _busy(pixels):
            # A halo keeps the text readable over fine hatching or dots.
            outline = (255, 255, 255) if np.dot(colour, _LUMA[::-1]) < 128 else (0, 0, 0)
    mask = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    patch = Patch(x0, y0, pixels, mask, colour, fill, centred, outline=outline)
    # With a halo, longer English can run on over the pattern, up to the
    # next text or panel border.
    size = block.line_size
    margin = int(size * 0.4)
    reach_x = int(max(0.6 * (x1 - x0), 4 * size))
    reach_y = int(max(1.4 * size, 0.5 * (y1 - y0)))
    left, right, down = _pattern_reach(frame, rect, repeats, reach_x, reach_y)
    patch.grow_left, patch.grow_right, patch.grow_down = (max(0, run - margin) for run in (left, right, down))
    return patch


def make_patch(frame: np.ndarray, block: Block, gray: np.ndarray | None = None, avoid: list[Rect] = ()) -> Patch:
    """Build the background patch that hides `block`'s original text.
    `avoid` are the rectangles of all text on screen (not to be copied)."""
    fh, fw = frame.shape[:2]
    size = block.line_size
    pad = max(2, int(round(size * 0.1)))
    x0, y0, x1, y1 = _padded_rect(block, fw, fh)
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
    if not uniform:
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        patch = _pattern_patch(frame, gray, block, list(avoid) or [(x0, y0, x1, y1)], centred)
        if patch is not None:
            return patch
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
    avoid = [_padded_rect(b, fw, fh) for b in blocks]  # translated or not
    blocks = [b for b in blocks if b.translation]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) if blocks else None
    patches = [make_patch(frame, b, gray, avoid) for b in blocks]
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
        flags = _WRAP | int(horizontal | vertical)
        if patch.outline is not None:
            painter.setPen(QColor(*patch.outline))
            radius = max(1.5, best_px * 0.12)
            for angle in range(0, 360, 30):
                offset = QPointF(radius * np.cos(np.radians(angle)), radius * np.sin(np.radians(angle)))
                painter.drawText(best.translated(offset), flags, text)
        painter.setPen(QColor(*patch.text_color))
        painter.drawText(best, flags, text)
    painter.end()
    return image
