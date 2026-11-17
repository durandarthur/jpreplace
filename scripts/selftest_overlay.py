"""End-to-end check of the real overlay window.

Shows a test "screenshot" in a window, puts the overlay on top of it, and
exercises one-shot and live translation through real screen capture, then
binding to that window (and following it when it moves) and full-screen
mode. Takes about 20 seconds and briefly puts windows on your primary screen.

Usage: uv run python scripts/selftest_overlay.py [out_dir]
"""

from __future__ import annotations

import ctypes
import faulthandler
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Slot
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QLabel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import make_samples  # noqa: E402

from jpreplace import app as overlay  # noqa: E402
from jpreplace.capture import ScreenGrabber  # noqa: E402
from jpreplace.config import Settings  # noqa: E402


class Relay(QObject):
    """Receives worker signals on the GUI thread (a bare lambda would run on
    the worker's thread)."""

    def __init__(self, on_ready, on_failed) -> None:
        super().__init__()
        self._on_ready, self._on_failed = on_ready, on_failed

    @Slot(str)
    def ready(self, _translator: str) -> None:
        self._on_ready()

    @Slot(str)
    def failed(self, message: str) -> None:
        self._on_failed(message)


def to_array(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGB32)
    return np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[:, :, :3].copy()


def main() -> int:
    faulthandler.dump_traceback_later(120, exit=True)  # never hang silently
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/selftest")
    out.mkdir(parents=True, exist_ok=True)
    qt = QApplication(sys.argv[:1])
    dpr = qt.primaryScreen().devicePixelRatio()
    origin = qt.primaryScreen().availableGeometry().topLeft()
    pages = {name: make() for name, make in make_samples.SAMPLES.items()}
    failures: list[str] = []
    timings: dict[str, float] = {}

    stage = QLabel()
    stage.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)

    def show_page(name: str) -> None:
        pixmap = QPixmap.fromImage(pages[name])
        pixmap.setDevicePixelRatio(dpr)
        stage.setPixmap(pixmap)
        stage.resize(pixmap.deviceIndependentSize().toSize())

    show_page("webpage")
    stage.move(origin.x() + 60, origin.y() + 80)
    stage.show()

    settings = Settings()
    settings.save = lambda: None  # do not touch the user's settings
    frame = overlay.Frame(settings)
    inner = stage.geometry()
    frame.setGeometry(
        inner.x() - overlay.BORDER,
        inner.y() - overlay.BORDER - overlay.BAR_HEIGHT,
        inner.width() + 2 * overlay.BORDER,
        inner.height() + 2 * overlay.BORDER + overlay.BAR_HEIGHT,
    )
    thread = QThread()
    worker = overlay.Worker(settings)
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
    # Whatever has the focus while this runs (a terminal, say) must not count
    # as another application covering the bound window.
    overlay.win32.foreground_window = lambda: int(stage.winId())

    def press_hotkey(hotkey_id: int) -> None:
        """Deliver a global hotkey the way Windows does (works while the frame is hidden)."""
        ctypes.windll.user32.PostMessageW(int(frame.winId()), 0x0312, hotkey_id, 0)

    def save_view(name: str) -> None:
        """What the user sees: the page with the overlay canvas on top."""
        page = pages[name].copy()
        painter = QPainter(page)
        if frame.canvas.image is not None:
            overlay_image = QImage(frame.canvas.image)
            overlay_image.setDevicePixelRatio(1.0)
            painter.drawImage(0, 0, overlay_image)
        painter.end()
        page.save(str(out / f"{name}.png"))

    def check_capture(name: str) -> None:
        """The overlay must be invisible to capture and pixel-aligned."""
        rect = frame.capture_rect()
        shot = ScreenGrabber("gdi").grab(*rect)
        expected = to_array(pages[name])
        if shot.shape != expected.shape:
            failures.append(f"{name}: captured {shot.shape}, expected {expected.shape}")
            return
        diff = float(np.abs(shot.astype(np.int16) - expected.astype(np.int16)).mean())
        print(f"{name}: capture differs from source by {diff:.2f} (0 = overlay invisible and aligned)")
        if diff > 1.0:
            failures.append(f"{name}: capture does not match the page (diff {diff:.2f})")

    def expect_overlay(name: str) -> None:
        image = frame.canvas.image
        if image is None or not to_array(image.convertToFormat(QImage.Format.Format_ARGB32)).any():
            failures.append(f"{name}: overlay is empty")
        print(f"{name}: status = {frame.status.text()!r}")

    marks: dict[str, float] = {}

    def step_translate() -> None:
        marks["one-shot"] = time.perf_counter()
        frame.translate_now()
        wait_for_image(lambda: step_after_translate())

    def wait_for_image(then, deadline: float = 20.0) -> None:
        start = time.perf_counter()

        def poll() -> None:
            if frame.canvas.image is not None and not frame.busy:
                then()
            elif time.perf_counter() - start > deadline:
                failures.append(f"timed out waiting for the overlay (status {frame.status.text()!r})")
                finish()
            else:
                QTimer.singleShot(20, poll)

        poll()

    def step_after_translate() -> None:
        timings["one-shot (web page)"] = time.perf_counter() - marks["one-shot"]
        expect_overlay("webpage")
        check_capture("webpage")
        save_view("webpage")
        # Live mode: change what is on screen and expect the overlay to follow.
        frame.clear()
        frame.live_button.setChecked(True)
        QTimer.singleShot(2500, step_live_switch)

    def step_live_switch() -> None:
        expect_overlay("webpage (live)")
        show_page("game")
        frame.canvas.set_image(None)
        marks["live"] = time.perf_counter()
        wait_for_image(step_live_done)

    def step_live_done() -> None:
        timings["live update (game)"] = time.perf_counter() - marks["live"]
        QTimer.singleShot(1500, step_live_final)

    def step_live_final() -> None:
        expect_overlay("game (live)")
        check_capture("game")
        save_view("game")
        frame.live_button.setChecked(False)
        # Translating an unchanged screen again must keep the overlay as it is.
        marks["image"] = frame.canvas.image
        marks["no-change"] = time.perf_counter()
        frame.translate_now()
        wait_idle(step_no_change)

    def wait_idle(then) -> None:
        if frame.busy:
            QTimer.singleShot(20, lambda: wait_idle(then))
        else:
            then()

    def step_no_change() -> None:
        timings["translate again, nothing changed"] = time.perf_counter() - marks["no-change"]
        if frame.canvas.image is not marks["image"] or "No change" not in frame.status.text():
            failures.append(f"unchanged screen was redrawn (status {frame.status.text()!r})")
        # Bind to the stage window, then move it: the overlay must follow.
        frame.bind_to(int(stage.winId()))  # covers the whole window: here, the same area as before
        if frame.bound != int(stage.winId()):
            failures.append(f"bound to {frame.bound}, expected the stage window {int(stage.winId())}")
            finish()
            return
        marks["rect"] = frame.capture_rect()
        stage.move(stage.x() + 40, stage.y() + 30)
        QTimer.singleShot(500, step_bind_moved)

    def step_bind_moved() -> None:
        x, y, w, h = marks["rect"]
        expected = (x + round(40 * dpr), y + round(30 * dpr), w, h)
        if frame.capture_rect() != expected:
            failures.append(f"bound overlay at {frame.capture_rect()}, expected {expected}")
        if frame.canvas.image is not marks["image"]:
            failures.append("overlay picture was dropped when following a window move")
        check_capture("game")
        # Full screen via the hotkey (the frame hides; the canvas covers the bound window).
        press_hotkey(overlay.HOTKEY_FULLSCREEN)
        QTimer.singleShot(400, step_fullscreen)

    def step_fullscreen() -> None:
        stage_rect = overlay.win32.client_rect(int(stage.winId()))
        if not frame.fullscreen or frame.isVisible() or frame.capture_rect() != stage_rect:
            failures.append(
                f"full screen: fullscreen={frame.fullscreen} frame visible={frame.isVisible()} "
                f"canvas={frame.capture_rect()} window={stage_rect}"
            )
        show_page("manga")
        frame.canvas.set_image(None)
        frame.request_forget.emit()
        frame.translate_now()
        wait_for_image(step_fullscreen_done)

    def step_fullscreen_done() -> None:
        expect_overlay("manga (full screen)")
        check_capture("manga")
        save_view("manga")
        press_hotkey(overlay.HOTKEY_FULLSCREEN)
        QTimer.singleShot(400, step_windowed)

    def step_windowed() -> None:
        if frame.fullscreen or not frame.isVisible():
            failures.append("did not leave full screen")
        frame.bind_button.click()  # bound: releases
        if frame.bound is not None:
            failures.append("did not unbind")
        finish()

    def finish() -> None:
        frame.close()
        stage.close()
        qt.quit()

    def watchdog() -> None:
        failures.append(f"gave up after 60 s (status: {frame.status.text()!r})")
        finish()

    QTimer.singleShot(60_000, watchdog)
    relay = Relay(
        lambda: QTimer.singleShot(300, step_translate),
        lambda message: failures.append(f"worker: {message}"),
    )
    worker.ready.connect(relay.ready)
    worker.failed.connect(relay.failed)
    thread.start()
    frame.show()
    qt.exec()
    thread.quit()
    thread.wait(5000)
    worker.pipeline.close()

    for name, seconds in timings.items():
        print(f"{name}: {seconds:.2f} s")
    for failure in failures:
        print("FAIL:", failure)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
