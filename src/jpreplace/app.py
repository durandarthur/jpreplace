"""The overlay: a movable, resizable frame whose interior shows the screen
underneath with Japanese text replaced by English."""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QObject, QPoint, QRect, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QRegion
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QWidget

from . import win32
from .capture import ScreenGrabber
from .config import Settings
from .layout import Block
from .pipeline import Pipeline, same_layout

BORDER = 5
BAR_HEIGHT = 30
ACCENT = QColor(66, 133, 244)
HOTKEY_TRANSLATE, HOTKEY_CLEAR, HOTKEY_LIVE = 1, 2, 3

_BAR_STYLE = """
QWidget#bar { background: rgba(24, 28, 38, 235); }
QLabel { color: #d8dce6; font: 12px 'Segoe UI'; }
QPushButton {
    color: #e8eaf0; background: rgba(255, 255, 255, 22); border: none;
    border-radius: 4px; padding: 3px 10px; font: 12px 'Segoe UI';
}
QPushButton:hover { background: rgba(255, 255, 255, 48); }
QPushButton:checked { background: rgb(66, 133, 244); }
QPushButton:disabled { color: #7c8290; }
QPushButton#close:hover { background: rgb(214, 62, 62); }
"""


class Worker(QObject):
    """Runs capture, OCR and translation off the GUI thread."""

    ready = Signal()
    status = Signal(str)
    failed = Signal(str)
    # overlay image (None = nothing changed), capture rect, summary text
    done = Signal(object, tuple, str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.pipeline = Pipeline(settings)
        self.grabber: ScreenGrabber | None = None
        self._last_frame: np.ndarray | None = None
        self._pending: list[Block] | None = None
        self._shown: list[Block] | None = None

    @Slot()
    def load(self) -> None:
        try:
            self.grabber = ScreenGrabber()
            self.pipeline.load(self.status.emit)
            self.ready.emit()
        except Exception as exc:  # surfaced in the title bar
            traceback.print_exc()
            self.failed.emit(f"Startup failed: {exc}")

    @Slot()
    def forget(self) -> None:
        """The overlay was cleared or moved; start from scratch next time."""
        self._last_frame = self._pending = self._shown = None

    @Slot(tuple)
    def translate(self, rect: tuple[int, int, int, int]) -> None:
        """One-shot: translate whatever is in `rect` right now."""
        try:
            start = time.perf_counter()
            frame = self.grabber.grab(*rect)
            result = self.pipeline.run(frame)
            self._last_frame, self._shown, self._pending = frame, result.blocks, None
            elapsed = time.perf_counter() - start
            self.done.emit(result.image, rect, _summary(len(result.blocks), elapsed))
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
            last = self._last_frame
            if last is not None and last.shape == frame.shape and np.array_equal(last, frame):
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
            self.done.emit(result.image, rect, _summary(len(blocks), elapsed))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))


def _summary(blocks: int, seconds: float) -> str:
    if not blocks:
        return "No Japanese text found"
    return f"{blocks} block{'s' if blocks != 1 else ''} · {seconds:.1f} s"


class Bar(QWidget):
    """Title bar; dragging it moves the frame."""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.window().windowHandle().startSystemMove()


class Canvas(QWidget):
    """Click-through window that paints the translated patches over the screen."""

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

    def set_image(self, image: QImage | None) -> None:
        self.image = image
        self.update()

    def paintEvent(self, _event) -> None:
        if self.image is not None:
            painter = QPainter(self)
            painter.drawImage(self.rect(), self.image)


class Frame(QWidget):
    """The visible frame: title bar plus border. Its interior is a hole."""

    request_translate = Signal(tuple)
    request_tick = Signal(tuple)
    request_forget = Signal()

    def __init__(self, settings: Settings) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.settings = settings
        self.setWindowTitle("jpreplace")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setMinimumSize(260, 140)
        self.canvas = Canvas(self)
        self.busy = True  # until models are loaded
        self.loaded = False

        bar = Bar(self)
        bar.setObjectName("bar")
        bar.setStyleSheet(_BAR_STYLE)
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        bar.setCursor(Qt.CursorShape.SizeAllCursor)
        self.bar = bar
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 3, 4, 3)
        row.setSpacing(6)
        self.status = QLabel("Starting…")
        self.translate_button = QPushButton("Translate")
        self.translate_button.setToolTip(f"Translate now ({settings.hotkey_translate})")
        self.live_button = QPushButton("Live")
        self.live_button.setCheckable(True)
        self.live_button.setToolTip(f"Keep translating as the screen changes ({settings.hotkey_live})")
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip(f"Show the original ({settings.hotkey_clear})")
        close = QPushButton("✕")
        close.setObjectName("close")
        row.addWidget(self.status, 1)
        for button in (self.translate_button, self.live_button, self.clear_button, close):
            button.setCursor(Qt.CursorShape.ArrowCursor)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.addWidget(button)
        self.translate_button.clicked.connect(self.translate_now)
        self.live_button.toggled.connect(self.set_live)
        self.clear_button.clicked.connect(self.clear)
        close.clicked.connect(self.close)
        self.translate_button.setEnabled(False)
        self.live_button.setEnabled(False)

        self.timer = QTimer(self)
        self.timer.setInterval(settings.live_interval_ms)
        self.timer.timeout.connect(self._tick)

        x, y, w, h = settings.geometry
        self.setGeometry(x, y, w, h)

    # -- geometry ----------------------------------------------------------

    def interior(self) -> QRect:
        return QRect(BORDER, BORDER + BAR_HEIGHT, self.width() - 2 * BORDER, self.height() - 2 * BORDER - BAR_HEIGHT)

    def _sync(self) -> None:
        self.bar.setGeometry(BORDER, BORDER, self.width() - 2 * BORDER, BAR_HEIGHT)
        # Cut the interior out of the window so clicks reach the app below.
        self.setMask(QRegion(self.rect()).subtracted(QRegion(self.interior())))
        inner = self.interior()
        self.canvas.setGeometry(QRect(self.mapToGlobal(inner.topLeft()), inner.size()))
        self._invalidate()

    def _invalidate(self) -> None:
        """The frame moved, so the picture no longer matches what is below."""
        if self.canvas.image is not None:
            self.canvas.set_image(None)
        self.request_forget.emit()

    def resizeEvent(self, _event) -> None:
        self._sync()

    def moveEvent(self, _event) -> None:
        self._sync()

    def showEvent(self, _event) -> None:
        self.canvas.show()
        self._sync()
        # The canvas must not show up in our own screen captures.
        if not win32.exclude_from_capture(int(self.canvas.winId())):
            self.set_status("Warning: cannot hide the overlay from screen capture")
        hwnd = int(self.winId())
        for hotkey_id, spec in (
            (HOTKEY_TRANSLATE, self.settings.hotkey_translate),
            (HOTKEY_CLEAR, self.settings.hotkey_clear),
            (HOTKEY_LIVE, self.settings.hotkey_live),
        ):
            try:
                win32.register_hotkey(hwnd, hotkey_id, spec)
            except ValueError as exc:
                self.set_status(str(exc))

    def closeEvent(self, _event) -> None:
        self.timer.stop()
        hwnd = int(self.winId())
        for hotkey_id in (HOTKEY_TRANSLATE, HOTKEY_CLEAR, HOTKEY_LIVE):
            win32.unregister_hotkey(hwnd, hotkey_id)
        geometry = self.geometry()
        self.settings.geometry = [geometry.x(), geometry.y(), geometry.width(), geometry.height()]
        self.settings.live = self.live_button.isChecked()
        self.settings.save()

    def capture_rect(self) -> tuple[int, int, int, int]:
        return win32.window_rect(int(self.canvas.winId()))

    # -- painting and mouse ------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), ACCENT)

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
        return False, 0

    # -- actions -----------------------------------------------------------

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    @Slot()
    def on_ready(self) -> None:
        self.loaded = True
        self.busy = False
        self.translate_button.setEnabled(True)
        self.live_button.setEnabled(True)
        self.set_status(f"Ready — {self.settings.hotkey_translate} to translate")
        if self.settings.live:
            self.live_button.setChecked(True)

    @Slot()
    def translate_now(self) -> None:
        if not self.loaded or self.busy:
            return
        self.busy = True
        self.canvas.set_image(None)
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
        if not self.loaded or self.busy:
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

    @Slot(object, tuple, str)
    def on_done(self, image: QImage | None, rect: tuple, summary: str) -> None:
        self.busy = False
        if rect != self.capture_rect():
            return  # the frame moved while this was being computed
        if image is not None:
            image.setDevicePixelRatio(self.canvas.devicePixelRatioF())
            self.canvas.set_image(image)
        if summary:
            self.set_status(("Live · " if self.timer.isActive() else "") + summary)

    @Slot(str)
    def on_failed(self, message: str) -> None:
        self.busy = not self.loaded
        self.set_status(f"Error: {message}")


def run(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    app.setQuitOnLastWindowClosed(True)
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
    thread.start()

    frame.show()
    code = app.exec()
    thread.quit()
    if not thread.wait(5_000):
        thread.terminate()  # still loading or mid-request; nothing to save
        thread.wait(2_000)
    worker.pipeline.close()
    return code
