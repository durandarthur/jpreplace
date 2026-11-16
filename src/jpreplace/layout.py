"""Group OCR lines into text blocks (paragraphs, labels, speech bubbles)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .ocr import Line

_JAPANESE = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]")
_ASCII_WORD = re.compile(r"[A-Za-z0-9]")
_MIN_WRAPPED_CHARS = 12  # a line shorter than this is a label, not a wrapped line

Box = tuple[float, float, float, float]


@dataclass
class Block:
    lines: list[Line]  # reading order
    text: str
    box: Box
    vertical: bool
    translation: str = ""

    @property
    def line_size(self) -> float:
        """Median glyph size: line height (horizontal) or column width (vertical)."""
        sizes = sorted(_thickness(ln) for ln in self.lines)
        return sizes[len(sizes) // 2]


def has_japanese(text: str) -> bool:
    return bool(_JAPANESE.search(text))


def _thickness(line: Line) -> float:
    x0, y0, x1, y1 = line.box
    return (x1 - x0) if line.vertical else (y1 - y0)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _same_paragraph(a: Line, b: Line) -> bool:
    if a.vertical != b.vertical:
        return False
    ta, tb = _thickness(a), _thickness(b)
    t = min(ta, tb)
    if max(ta, tb) > 1.4 * t:
        return False
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box

    if a.vertical:
        # Columns side by side, read right to left.
        gap = max(ax0, bx0) - min(ax1, bx1)
        if not -0.5 * t <= gap <= 0.9 * t:
            return False
        return _overlap(ay0, ay1, by0, by1) >= 0.5 * min(ay1 - ay0, by1 - by0)

    # Fragments of the same row. Japanese has no spaces, so a visible gap
    # means separate items (menu entries, table cells): only join boxes that
    # overlap, i.e. whose text is closer than the padding the detector adds.
    if _overlap(ay0, ay1, by0, by1) >= 0.7 * t:
        gap = max(ax0, bx0) - min(ax1, bx1)
        return gap <= -0.2 * t

    # Consecutive rows: `a` is the upper one.
    upper = a
    if ay0 > by0:
        upper = b
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = (bx0, by0, bx1, by1), (ax0, ay0, ax1, ay1)
    gap = by0 - ay1
    if not -0.5 * t <= gap <= 0.9 * t:
        return False
    left = abs(ax0 - bx0) <= t
    centre = abs((ax0 + ax1) - (bx0 + bx1)) / 2 <= t
    if centre and not left:
        return True  # centred lines of different widths: a caption or bubble
    # Left-aligned wrapped text: every line but the last runs to the margin.
    # Lists and menus look the same geometrically, but their items are short,
    # and merging those is worse than splitting a narrow paragraph.
    return left and ax1 >= bx1 - t and len(upper.text) >= _MIN_WRAPPED_CHARS


def _join(parts: list[str]) -> str:
    out = ""
    for part in parts:
        if out and _ASCII_WORD.match(out[-1]) and _ASCII_WORD.match(part[0]):
            out += " "
        out += part
    return out


def _make_block(lines: list[Line]) -> Block:
    vertical = lines[0].vertical
    if vertical:
        ordered = sorted(lines, key=lambda ln: -ln.box[0])
    else:
        # Sort into rows, then left to right within a row.
        ordered = sorted(lines, key=lambda ln: ln.box[1])
        rows: list[list[Line]] = []
        for ln in ordered:
            if rows and _overlap(*rows[-1][0].box[1::2], *ln.box[1::2]) >= 0.5 * _thickness(ln):
                rows[-1].append(ln)
            else:
                rows.append([ln])
        ordered = [ln for row in rows for ln in sorted(row, key=lambda ln: ln.box[0])]
    boxes = [ln.box for ln in ordered]
    box = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
    return Block(ordered, _join([ln.text for ln in ordered]), box, vertical)


def group_lines(lines: list[Line]) -> list[Block]:
    """Cluster lines into blocks, returned in reading order (top to bottom)."""
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if find(i) != find(j) and _same_paragraph(lines[i], lines[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[Line]] = {}
    for i, line in enumerate(lines):
        groups.setdefault(find(i), []).append(line)
    blocks = [_make_block(g) for g in groups.values()]
    blocks.sort(key=lambda b: (round(b.box[1] / 24), b.box[0]))
    return blocks
