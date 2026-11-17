"""The overlay: a movable, resizable frame whose interior shows the screen
underneath with Japanese text replaced by English."""

from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QObject, QPoint, QRect, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QRegion
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy, QWidget

from . import win32
from .capture import ScreenGrabber
from .config import Settings
from .layout import Block
from .pipeline import Pipeline, same_layout

BORDER = 5
BAR_HEIGHT = 30
ACCENT = QColor(66, 133, 244)
BOUND_ACCENT = QColor(52, 168, 83)
HOTKEY_TRANSLATE, HOTKEY_CLEAR, HOTKEY_LIVE, HOTKEY_FULLSCREEN = 1, 2, 3, 4
HOTKEY_CANCEL = 5  # Esc, only while picking a window
FOLLOW_MS = 15  # how often a bound window's position is checked
TOAST_MS = 2500
BIND_TIP = "Attach the frame to a window: it covers that window and follows it around"

_BAR_STYLE = """
QWidget#bar { background: rgba(24, 28, 38, 235); }
QLabel { color: #d8dce6; font: 12px 'Segoe UI'; }
QPushButton {
    color: #e8eaf0; background: rgba(255, 255, 255, 22); border: none;
    border-radius: 4px; padding: 3px 8px; font: 12px 'Segoe UI';
}
QPushButton:hover { background: rgba(255, 255, 255, 48); }
QPushButton:checked { background: rgb(66, 133, 244); }
QPushButton#bind:checked { background: rgb(52, 168, 83); }
QPushButton:disabled { color: #7c8290; }
QPushButton#close:hover { background: rgb(214, 62, 62); }
"""


class Worker(QObject):
    """Runs capture, OCR and translation off the GUI thread."""

    ready = Signal(str)  # translator name
    status = Signal(str)
    failed = Signal(str)
    # overlay image (None = keep what is shown), capture rect, summary text
    done = Signal(object, tuple, str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.pipeline = Pipeline(settings)
        self.grabber: ScreenGrabber | None = None
        self._last_frame: np.ndarray | None = None
        self._pending: list[Block] | None = None
        self._shown: list[Block] | None = None

    @Slot()
    def load(self) -> None:
        """Load whatever is not loaded yet; also how the translator is
        restarted after the settings change."""
        try:
            if self.grabber is None:
                self.grabber = ScreenGrabber(self.settings.capture)
            if self.pipeline.translator is not None:
                self.pipeline.load_translator(self.status.emit)
            else:
                self.pipeline.load(self.status.emit)
            self.forget()
            self.ready.emit(self.pipeline.translator_name)
        except Exception as exc:  # surfaced in the title bar
            traceback.print_exc()
            stage = "Translator failed" if self.pipeline.ocr is not None else "Startup failed"
            self.failed.emit(f"{stage}: {exc}")

    @Slot(tuple)
    def prepare(self, rect: tuple[int, int, int, int]) -> None:
        """The frame was resized: tune the detector for the new size now."""
        try:
            self.pipeline.prepare(rect[3], rect[2])
        except Exception:
            traceback.print_exc()

    @Slot()
    def forget(self) -> None:
        """The overlay was cleared or moved; start from scratch next time."""
        self._last_frame = self._pending = self._shown = None

    def _unchanged(self, frame: np.ndarray) -> bool:
        last = self._last_frame
        return last is not None and last.shape == frame.shape and np.array_equal(last, frame)

    @Slot(tuple)
    def translate(self, rect: tuple[int, int, int, int]) -> None:
        """One-shot: translate whatever is in `rect` right now. If the text
        on screen is what the overlay already shows, it is left as it is."""
        try:
            start = time.perf_counter()
            frame = self.grabber.grab(*rect)
            if self._shown is not None and self._unchanged(frame):
                self.done.emit(None, rect, "No change")
                return
            blocks = self.pipeline.read(frame)
            self._last_frame = frame
            if self._shown is not None and same_layout(blocks, self._shown):
                self.done.emit(None, rect, "No change")
                return
            result = self.pipeline.run(frame, blocks)
            self._shown, self._pending = result.blocks, None
            elapsed = time.perf_counter() - start
            self.done.emit(result.image, rect, _summary(len(result.blocks), elapsed, result.error))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))

    @Slot(tuple)
    def tick(self, rect: tuple[int, int, int, int]) -> None:
        """Live mode: re-translate only when the text on screen has changed
        and has stopped changing."""
        try:
            start = time.perf_counter()
            frame = self.grabber.grab(*rect)
            if self._unchanged(frame):
                # Screen is static. Finish a translation that was waiting for it.
                if self._pending is None:
                    self.done.emit(None, rect, "")
                    return
                blocks = self._pending
            else:
                self._last_frame = frame
                blocks = self.pipeline.read(frame)

            if self._shown is not None and same_layout(blocks, self._shown):
                self._pending = None
                self.done.emit(None, rect, "")
                return

            missing = self.pipeline.fill_cached(blocks)
            settled = self._pending is not None and same_layout(blocks, self._pending)
            if missing and not settled:
                # Text just changed (scrolling, typewriter effect…): show what
                # is already known and wait one tick for it to settle.
                self._pending, self._shown = blocks, None
                self.done.emit(self.pipeline.draw(frame, blocks), rect, "")
                return

            result = self.pipeline.run(frame, blocks)
            self._pending, self._shown = None, blocks
            elapsed = time.perf_counter() - start
            self.done.emit(result.image, rect, _summary(len(blocks), elapsed, result.error))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))


def _summary(blocks: int, seconds: float, error: str | None = None) -> str:
    if not blocks:
        return "No Japanese text found"
    text = f"{blocks} block{'s' if blocks != 1 else ''} · {seconds:.1f} s"
    if error:
        text += f" · some text not translated: {error[:120]}"
    return text


class Bar(QWidget):
    """Title bar; dragging it moves the frame."""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.window().windowHandle().startSystemMove()


class Canvas(QWidget):
    """Click-through window that paints the translated patches over the
    screen, plus short status messages while the frame is hidden."""

    def __init__(self, owner: QWidget) -> None:
        super().__init__(
            owner,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.image: QImage | None = None
        self.toast_text = ""
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(lambda: self.toast(""))

    def set_image(self, image: QImage | None) -> None:
        self.image = image
        self.update()

    def toast(self, text: str) -> None:
        self.toast_text = text
        if text:
            self._toast_timer.start(TOAST_MS)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        if self.image is not None:
            painter.drawImage(self.rect(), self.image)
        if self.toast_text:
            font = QFont("Segoe UI")
            font.setPixelSize(13)
            painter.setFont(font)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            box = painter.fontMetrics().boundingRect(self.toast_text).adjusted(-10, -6, 10, 6)
            box.moveTopRight(QPoint(self.width() - 16, 16))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(24, 28, 38, 220))
            painter.drawRoundedRect(QRectF(box), 6, 6)
            painter.setPen(QColor(232, 234, 240))
            painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), self.toast_text)


class WindowPicker(QWidget):
    """Highlights the window under the mouse until one is clicked. The click
    is caught before the application gets it. `picked` carries the window,
    or 0 if cancelled (right click, Esc)."""

    picked = Signal(int)

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.hwnd: int | None = None
        self.title = ""
        self._catcher: win32.ClickCatcher | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._hover)

    def start(self) -> None:
        self.stop()
        self._catcher = win32.ClickCatcher(self._clicked)
        self.hwnd = None
        self._timer.start()
        self._hover()

    def stop(self) -> None:
        if self._catcher is not None:
            self._catcher.remove()
            self._catcher = None
        self._timer.stop()
        self.hide()

    def cancel(self) -> None:
        if self._catcher is not None:
            self.stop()
            self.picked.emit(0)

    def _hover(self) -> None:
        hwnd = win32.window_at(*win32.cursor_pos())
        if hwnd is None:
            self.hwnd = None
            self.hide()
            return
        try:
            rect = win32.client_rect(hwnd)
        except OSError:
            return
        if hwnd != self.hwnd:
            self.hwnd, self.title = hwnd, win32.window_title(hwnd) or "untitled window"
            self.update()
        if not self.isVisible():
            self.show()
        if win32.window_rect(int(self.winId())) != rect:
            win32.move_window(int(self.winId()), rect)

    def _clicked(self, button: str, x: int, y: int) -> None:
        # Called from inside the mouse hook: do the work afterwards.
        QTimer.singleShot(0, lambda: self._finish(button == "left", x, y))

    def _finish(self, accept: bool, x: int, y: int) -> None:
        if self._catcher is None:
            return
        self.stop()
        self.picked.emit((win32.window_at(x, y) or 0) if accept else 0)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(52, 168, 83, 45))
        painter.setPen(QPen(BOUND_ACCENT, 6))
        painter.drawRect(self.rect().adjusted(3, 3, -3, -3))
        font = QFont("Segoe UI")
        font.setPixelSize(14)
        painter.setFont(font)
        text = f"Bind to “{self.title}” — click   (right-click or Esc to cancel)"
        box = painter.fontMetrics().boundingRect(text).adjusted(-10, -6, 10, 6)
        box.moveTopLeft(QPoint(14, 14))
        painter.fillRect(box, BOUND_ACCENT)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)


class Frame(QWidget):
    """The visible frame: title bar plus border. Its interior is a hole."""

    request_translate = Signal(tuple)
    request_tick = Signal(tuple)
    request_forget = Signal()
    request_prepare = Signal(tuple)
    request_reload = Signal()

    def __init__(self, settings: Settings) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.settings = settings
        self.setWindowTitle("jpreplace")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setMinimumSize(360, 140)
        self.canvas = Canvas(self)
        self.busy = True  # until models are loaded
        self.loaded = False
        self._setup_done = False
        self._one_shot = False  # the request in flight is a one-shot translation

        # Bound window (see bind()) and full-screen mode.
        self.bound: int | None = None
        self.bound_pid = 0
        self.bound_title = ""
        self._fractions = (0.0, 0.0, 1.0, 1.0)  # interior within the bound window's client area
        self._target_rect: win32.Rect | None = None  # bound window's client area, last seen
        self._placing_until = 0.0  # moves before this time are our own (following the window)
        self._last_capture: win32.Rect | None = None
        self.fullscreen = False
        self._windowed_geometry: QRect | None = None
        self.hidden_by_follow = False
        self._last_raise = 0.0

        bar = Bar(self)
        bar.setObjectName("bar")
        bar.setStyleSheet(_BAR_STYLE)
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        bar.setCursor(Qt.CursorShape.SizeAllCursor)
        self.bar = bar
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 3, 4, 3)
        row.setSpacing(4)
        self.status = QLabel("Starting…")
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.translate_button = QPushButton("Translate")
        self.translate_button.setToolTip(f"Translate now ({settings.hotkey_translate})")
        self.live_button = QPushButton("Live")
        self.live_button.setCheckable(True)
        self.live_button.setToolTip(f"Keep translating as the screen changes ({settings.hotkey_live})")
        self.bind_button = QPushButton("Bind")
        self.bind_button.setObjectName("bind")
        self.bind_button.setCheckable(True)
        self.bind_button.setToolTip(BIND_TIP)
        self.picker = WindowPicker()
        self.picker.picked.connect(self._picked)
        self.fullscreen_button = QPushButton("⛶")
        self.fullscreen_button.setCheckable(True)
        self.fullscreen_button.setToolTip(
            "Full screen: cover the bound window, or the whole monitor, with no frame "
            f"({settings.hotkey_fullscreen} to leave)"
        )
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip(f"Show the original ({settings.hotkey_clear})")
        self.settings_button = QPushButton("⚙")
        self.settings_button.setToolTip("Translation model settings")
        close = QPushButton("✕")
        close.setObjectName("close")
        row.addWidget(self.status, 1)
        for button in (
            self.translate_button, self.live_button, self.bind_button, self.fullscreen_button,
            self.clear_button, self.settings_button, close,
        ):
            button.setCursor(Qt.CursorShape.ArrowCursor)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.addWidget(button)
        self.translate_button.clicked.connect(self.translate_now)
        self.live_button.toggled.connect(self.set_live)
        self.bind_button.clicked.connect(self._bind_clicked)
        self.fullscreen_button.toggled.connect(self.set_fullscreen)
        self.clear_button.clicked.connect(self.clear)
        self.settings_button.clicked.connect(self.open_settings)
        close.clicked.connect(self.close)
        self.translate_button.setEnabled(False)
        self.live_button.setEnabled(False)

        self.timer = QTimer(self)
        self.timer.setInterval(settings.live_interval_ms)
        self.timer.timeout.connect(self._tick)
        # After a resize settles, let the worker get ready for the new size.
        self.resized = QTimer(self)
        self.resized.setSingleShot(True)
        self.resized.setInterval(400)
        self.resized.timeout.connect(self._prepare)
        self.follower = QTimer(self)
        self.follower.setTimerType(Qt.TimerType.PreciseTimer)
        self.follower.setInterval(FOLLOW_MS)
        self.follower.timeout.connect(self._follow)

        x, y, w, h = settings.geometry
        self.setGeometry(x, y, w, h)

    # -- geometry ----------------------------------------------------------

    def interior(self) -> QRect:
        return QRect(BORDER, BORDER + BAR_HEIGHT, self.width() - 2 * BORDER, self.height() - 2 * BORDER - BAR_HEIGHT)

    def _sync(self) -> None:
        self.bar.setGeometry(BORDER, BORDER, self.width() - 2 * BORDER, BAR_HEIGHT)
        # Cut the interior out of the window so clicks reach the app below.
        self.setMask(QRegion(self.rect()).subtracted(QRegion(self.interior())))
        if self.fullscreen:
            return  # the canvas covers the screen, not the frame's interior
        inner = self.interior()
        self.canvas.setGeometry(QRect(self.mapToGlobal(inner.topLeft()), inner.size()))
        self._canvas_moved()

    def _canvas_moved(self) -> None:
        """The canvas has a new position or size."""
        rect = self.capture_rect()
        before, self._last_capture = self._last_capture, rect
        if rect == before:
            return
        # Qt may report our own moves a little later, and a pixel off at
        # fractional display scaling.
        ours = time.monotonic() < self._placing_until
        if ours and before is not None and rect[2:] == before[2:]:
            # Moved along with the bound window, so the content underneath
            # moved too: the picture is still right.
            return
        if self.bound and not ours and not self.fullscreen:
            self._remember_fractions(rect)  # the user adjusted the frame over the window
        self._invalidate()

    def _invalidate(self) -> None:
        """The picture no longer matches what is below."""
        if self.canvas.image is not None:
            self.canvas.set_image(None)
        self.request_forget.emit()

    def resizeEvent(self, _event) -> None:
        self._sync()
        self.resized.start()

    def _prepare(self) -> None:
        if self.loaded and self.overlay_visible():
            self.request_prepare.emit(self.capture_rect())

    def moveEvent(self, _event) -> None:
        self._sync()

    def showEvent(self, _event) -> None:
        self.canvas.show()
        self._sync()
        if self._setup_done:
            return
        self._setup_done = True
        # The canvas must not show up in our own screen captures.
        if not win32.exclude_from_capture(int(self.canvas.winId())):
            self.set_status("Warning: cannot hide the overlay from screen capture")
        hwnd = int(self.winId())
        for hotkey_id, spec in (
            (HOTKEY_TRANSLATE, self.settings.hotkey_translate),
            (HOTKEY_CLEAR, self.settings.hotkey_clear),
            (HOTKEY_LIVE, self.settings.hotkey_live),
            (HOTKEY_FULLSCREEN, self.settings.hotkey_fullscreen),
        ):
            try:
                win32.register_hotkey(hwnd, hotkey_id, spec)
            except ValueError as exc:
                self.set_status(str(exc))

    def closeEvent(self, _event) -> None:
        self.timer.stop()
        self.follower.stop()
        self.picker.stop()
        self.picker.close()
        hwnd = int(self.winId())
        for hotkey_id in (HOTKEY_TRANSLATE, HOTKEY_CLEAR, HOTKEY_LIVE, HOTKEY_FULLSCREEN):
            win32.unregister_hotkey(hwnd, hotkey_id)
        geometry = self._windowed_geometry if self.fullscreen and self._windowed_geometry else self.geometry()
        self.settings.geometry = [geometry.x(), geometry.y(), geometry.width(), geometry.height()]
        self.settings.live = self.live_button.isChecked()
        self.settings.save()
        self.canvas.close()

    def capture_rect(self) -> tuple[int, int, int, int]:
        return win32.window_rect(int(self.canvas.winId()))

    def overlay_visible(self) -> bool:
        return self.canvas.isVisible() and not self.hidden_by_follow

    # -- painting and mouse ------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BOUND_ACCENT if self.bound else ACCENT)

    def _edges(self, pos: QPoint) -> Qt.Edge:
        grip = 14  # corners are easier to grab than the thin border
        edges = Qt.Edge(0)
        on_x = pos.x() < BORDER or pos.x() >= self.width() - BORDER
        on_y = pos.y() < BORDER or pos.y() >= self.height() - BORDER
        if pos.x() < (grip if on_y else BORDER):
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= self.width() - (grip if on_y else BORDER):
            edges |= Qt.Edge.RightEdge
        if pos.y() < (grip if on_x else BORDER):
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= self.height() - (grip if on_x else BORDER):
            edges |= Qt.Edge.BottomEdge
        return edges

    def mouseMoveEvent(self, event) -> None:
        edges = self._edges(event.position().toPoint())
        horizontal = bool(edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge))
        vertical = bool(edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge))
        if horizontal and vertical:
            falling = edges in (Qt.Edge.LeftEdge | Qt.Edge.TopEdge, Qt.Edge.RightEdge | Qt.Edge.BottomEdge)
            self.setCursor(Qt.CursorShape.SizeFDiagCursor if falling else Qt.CursorShape.SizeBDiagCursor)
        elif horizontal:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif vertical:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            edges = self._edges(event.position().toPoint())
            if edges:
                self.windowHandle().startSystemResize(edges)

    def nativeEvent(self, event_type, message):
        if event_type == b"windows_generic_MSG":
            hotkey = win32.hotkey_id_from_message(int(message))
            if hotkey == HOTKEY_TRANSLATE:
                self.translate_now()
            elif hotkey == HOTKEY_CLEAR:
                self.clear()
            elif hotkey == HOTKEY_LIVE:
                self.live_button.toggle()
            elif hotkey == HOTKEY_FULLSCREEN:
                self.fullscreen_button.toggle()
            elif hotkey == HOTKEY_CANCEL:
                self.picker.cancel()
        return False, 0

    # -- binding to a window -------------------------------------------------

    @Slot(bool)
    def _bind_clicked(self, checked: bool) -> None:
        """Bound: release. Otherwise offer the ways to choose a window."""
        if not checked:
            self._unbind("Unbound")
            return
        self._set_bind_checked(False)  # until a window is chosen
        menu = QMenu(self)
        menu.addAction("Click a window…", self.start_picker)
        menu.addAction("Window under the frame (keep this area)", self.bind_under_frame)
        menu.addSeparator()
        windows = win32.list_windows()[:30]
        for hwnd, title, program in windows:
            label = title if len(title) <= 60 else title[:57] + "…"
            action = menu.addAction(f"{label}   ·   {program}" if program else label)
            action.triggered.connect(lambda _=False, h=hwnd: self.bind_to(h))
        if not windows:
            menu.addAction("No other windows").setEnabled(False)
        menu.exec(self.bind_button.mapToGlobal(self.bind_button.rect().bottomLeft()))

    def _set_bind_checked(self, checked: bool) -> None:
        self.bind_button.blockSignals(True)
        self.bind_button.setChecked(checked)
        self.bind_button.blockSignals(False)

    @Slot()
    def start_picker(self) -> None:
        try:
            win32.register_hotkey(int(self.winId()), HOTKEY_CANCEL, "esc")
            self.picker.start()
        except OSError as exc:
            win32.unregister_hotkey(int(self.winId()), HOTKEY_CANCEL)
            self.set_status(f"Cannot pick a window: {exc}")
            return
        self.set_status("Click the window to bind (right-click or Esc to cancel)")

    @Slot(int)
    def _picked(self, hwnd: int) -> None:
        win32.unregister_hotkey(int(self.winId()), HOTKEY_CANCEL)
        if hwnd:
            self.bind_to(hwnd)
        else:
            self.set_status("Ready" if self.loaded else self.status.text())

    @Slot()
    def bind_under_frame(self) -> None:
        x, y, w, h = self.capture_rect()
        hwnd = win32.window_at(x + w // 2, y + h // 2)
        if hwnd is None:
            self.set_status("No window under the frame to bind to")
            return
        self.bind_to(hwnd, cover=False)

    def bind_to(self, hwnd: int, cover: bool = True) -> None:
        """Follow `hwnd` from now on. With `cover`, the frame moves over the
        whole of its content; otherwise it keeps the part it covers now."""
        if self.bound is not None:
            self._unbind("")
        if not win32.is_window(hwnd):
            self.set_status("That window has closed")
            return
        if cover:
            win32.bring_to_front(hwnd)
        self.bound, self.bound_pid = hwnd, win32.window_pid(hwnd)
        self.bound_title = win32.window_title(hwnd) or "untitled window"
        self._target_rect = win32.client_rect(hwnd)
        if cover:
            self._fractions = (0.0, 0.0, 1.0, 1.0)
        else:
            self._remember_fractions(self.capture_rect())
        self._set_bind_checked(True)
        self.bind_button.setToolTip(f"Bound to “{self.bound_title}” — click to release")
        self.set_status(f"Bound to “{self.bound_title}”")
        self.update()
        if (cover or self.fullscreen) and self._target_rect[2] > 0 and self._target_rect[3] > 0:
            self._place(self._target_rect)
        self.follower.start()

    def _unbind(self, message: str) -> None:
        was_bound = self.bound is not None
        self.bound = None
        self._target_rect = None
        self.bind_button.setToolTip(BIND_TIP)
        self._set_bind_checked(False)
        self._show_overlay(True)
        if self.fullscreen:
            self._place(win32.monitor_rect(int(self.canvas.winId())))
        else:
            self.follower.stop()
        self.update()
        if was_bound and message:
            self.set_status(message)

    def _remember_fractions(self, interior: win32.Rect) -> None:
        tx, ty, tw, th = self._target_rect
        x, y, w, h = interior
        self._fractions = ((x - tx) / tw, (y - ty) / th, (x + w - tx) / tw, (y + h - ty) / th)

    def _follow(self) -> None:
        """Keep up with the bound window: same place, shown only when it is."""
        if self.fullscreen and time.monotonic() - self._last_raise > 1.0:
            # Full-screen games like to put themselves on top.
            self._last_raise = time.monotonic()
            win32.raise_topmost(int(self.canvas.winId()))
        if self.bound is None:
            return
        try:
            visible = win32.is_on_screen(self.bound)
            rect = win32.client_rect(self.bound) if visible else self._target_rect
        except OSError:
            visible = False
        if not win32.is_window(self.bound):
            self._unbind("The bound window was closed")
            return
        if visible and rect[2] > 0 and rect[3] > 0:
            front = win32.foreground_window()
            if front and front != self.bound and win32.window_pid(front) not in (self.bound_pid, os.getpid()):
                # Another application is in front: hide if it covers the window.
                try:
                    visible = win32.is_shell(front) or not win32.overlaps(win32.window_rect(front), rect)
                except OSError:
                    pass
        else:
            visible = False
        self._show_overlay(visible)
        if visible and rect != self._target_rect:
            self._target_rect = rect
            self._place(rect)

    def _place(self, target: win32.Rect) -> None:
        """Put the canvas over `target` (physical pixels): all of it in full
        screen, else the remembered part of it, with the frame around."""
        if self.fullscreen:
            interior = target
        else:
            tx, ty, tw, th = target
            fx0, fy0, fx1, fy1 = self._fractions
            x0, y0 = round(tx + fx0 * tw), round(ty + fy0 * th)
            interior = (x0, y0, round(tx + fx1 * tw) - x0, round(ty + fy1 * th) - y0)
        self._placing_until = time.monotonic() + 0.3
        if self.fullscreen:
            win32.move_window(int(self.canvas.winId()), interior)
            self._canvas_moved()
            return
        # The frame's border around the canvas, in physical pixels.
        scale = self.devicePixelRatioF()
        side, top = round(BORDER * scale), round((BORDER + BAR_HEIGHT) * scale)
        x, y, w, h = interior
        # Keep the title bar on screen (a maximised window's content starts
        # right below the top of the monitor): cover a little less instead.
        screen_top = win32.monitor_rect(self.bound or int(self.winId()))[1]
        if y - top < screen_top and h - (screen_top - (y - top)) > 4 * top:
            cut = screen_top - (y - top)
            y, h = y + cut, h - cut
        win32.move_window(int(self.winId()), (x - side, y - top, w + 2 * side, h + top + side))

    def _show_overlay(self, visible: bool) -> None:
        if visible == (not self.hidden_by_follow):
            return
        self.hidden_by_follow = not visible
        if visible:
            if not self.fullscreen:
                self.show()
            self.canvas.show()
        else:
            self.canvas.hide()
            self.hide()

    # -- full screen ---------------------------------------------------------

    @Slot(bool)
    def set_fullscreen(self, enabled: bool) -> None:
        if enabled == self.fullscreen:
            return
        if enabled:
            self._windowed_geometry = self.geometry()
            target = self._target_rect if self.bound else win32.monitor_rect(int(self.canvas.winId()))
            self.fullscreen = True
            self.hide()
            self.canvas.show()
            self._place(target)
            self.follower.start()
            self.set_status(f"Full screen — {self.settings.hotkey_fullscreen} to leave")
        else:
            self.fullscreen = False
            if not self.bound:
                self.follower.stop()
            if not self.hidden_by_follow:
                self.show()
            if self.bound and self._target_rect:
                self._place(self._target_rect)
            elif self._windowed_geometry is not None:
                self.setGeometry(self._windowed_geometry)
            self._sync()
            self.canvas.toast("")
            self.set_status("Ready")

    # -- actions -----------------------------------------------------------

    def set_status(self, text: str) -> None:
        self.status.setText(text)
        self.status.setToolTip(text)
        if self.fullscreen:
            self.canvas.toast(text)

    @Slot(str)
    def on_ready(self, translator: str) -> None:
        self.loaded = True
        self.busy = False
        self.translate_button.setEnabled(True)
        self.live_button.setEnabled(True)
        self.set_status(f"Ready ({translator}) — {self.settings.hotkey_translate} to translate")
        self._prepare()
        if self.settings.live:
            self.live_button.setChecked(True)

    @Slot()
    def translate_now(self) -> None:
        if not self.loaded or self.busy or not self.overlay_visible():
            return
        self.busy = self._one_shot = True
        self.set_status("Translating…")
        self.request_translate.emit(self.capture_rect())

    @Slot(bool)
    def set_live(self, enabled: bool) -> None:
        if enabled:
            self.request_forget.emit()
            self.timer.start()
            self.set_status("Live")
            self._tick()
        else:
            self.timer.stop()
            self.set_status("Ready")

    def _tick(self) -> None:
        if not self.loaded or self.busy or not self.overlay_visible():
            return
        self.busy = True
        self.request_tick.emit(self.capture_rect())

    @Slot()
    def clear(self) -> None:
        if self.live_button.isChecked():
            self.live_button.setChecked(False)
        self.canvas.set_image(None)
        self.request_forget.emit()
        if self.loaded:
            self.set_status("Ready")

    @Slot()
    def open_settings(self) -> None:
        from .settings_dialog import SettingsDialog

        dialog = SettingsDialog(self.settings, self)
        if not dialog.exec() or not dialog.translator_changed:
            return
        self.settings.save()
        # Restart translation with the new model.
        self.settings.live = self.live_button.isChecked()
        self.live_button.setChecked(False)
        self.canvas.set_image(None)
        self.loaded, self.busy = False, True
        self.translate_button.setEnabled(False)
        self.live_button.setEnabled(False)
        self.set_status("Loading translator…")
        self.request_reload.emit()

    @Slot(object, tuple, str)
    def on_done(self, image: QImage | None, rect: tuple, summary: str) -> None:
        self.busy = False
        one_shot, self._one_shot = self._one_shot, False
        if rect != self.capture_rect():
            # The frame moved while this was being computed. Live mode will
            # catch up by itself; a one-shot request is redone.
            if one_shot:
                self.translate_now()
            return
        if image is not None:
            image.setDevicePixelRatio(self.canvas.devicePixelRatioF())
            self.canvas.set_image(image)
        if summary:
            self.set_status(("Live · " if self.timer.isActive() else "") + summary)

    @Slot(str)
    def on_failed(self, message: str) -> None:
        self.busy = not self.loaded
        self._one_shot = False
        self.set_status(f"Error: {message}")


def run(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    app.setQuitOnLastWindowClosed(False)  # the frame hides itself in full screen and when bound
    settings = Settings.load()
    frame = Frame(settings)

    thread = QThread()
    worker = Worker(settings)
    worker.moveToThread(thread)
    thread.started.connect(worker.load)
    worker.status.connect(frame.set_status)
    worker.ready.connect(frame.on_ready)
    worker.failed.connect(frame.on_failed)
    worker.done.connect(frame.on_done)
    frame.request_translate.connect(worker.translate)
    frame.request_tick.connect(worker.tick)
    frame.request_forget.connect(worker.forget)
    frame.request_prepare.connect(worker.prepare)
    frame.request_reload.connect(worker.load)
    frame.destroyed.connect(app.quit)
    frame.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    thread.start()

    frame.show()
    code = app.exec()
    thread.quit()
    if not thread.wait(5_000):
        thread.terminate()  # still loading or mid-request; nothing to save
        thread.wait(2_000)
    worker.pipeline.close()
    return code
