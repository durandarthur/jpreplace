"""End-to-end check of the real overlay window.

Shows a test "screenshot" in a window, puts the overlay on top of it, and
exercises one-shot and live translation through real screen capture. Takes
about 15 seconds and briefly puts windows on your primary screen.

Usage: uv run python scripts/selftest_overlay.py [out_dir]
"""

from __future__ import annotations

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

    @Slot()
    def ready(self) -> None:
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
        shot = ScreenGrabber().grab(*rect)
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
                failures.append("timed out waiting for the overlay")
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
