"""Screen capture of a rectangle given in physical pixels.

DXGI desktop duplication (via dxcam) sees what is actually on the monitor,
including full-screen games and hardware-accelerated video that GDI capture
can miss or show as black. GDI (mss) is the fallback: for rectangles that
span monitors, when duplication is unavailable (remote desktop, some
hybrid-GPU laptops) or after it is lost (display mode change, UAC prompt).
Both honour the overlay's exclusion from capture."""

from __future__ import annotations

import time
import traceback

import mss
import numpy as np

_RETRY_DXGI_AFTER = 5.0  # seconds before trying duplication again after a failure


class _Output:
    def __init__(self, camera, left: int, top: int, right: int, bottom: int) -> None:
        self.camera = camera
        self.left, self.top, self.right, self.bottom = left, top, right, bottom
        self.region: tuple[int, int, int, int] | None = None
        self.frame: np.ndarray | None = None  # last frame of `region`

    def contains(self, left: int, top: int, width: int, height: int) -> bool:
        return self.left <= left and self.top <= top and left + width <= self.right and top + height <= self.bottom


class ScreenGrabber:
    """Must be created and used on the same thread. `method` is "auto",
    "dxgi" or "gdi"."""

    def __init__(self, method: str = "auto") -> None:
        self._sct = mss.mss()
        self.method = method
        self._outputs: list[_Output] | None = None
        self._failed_at = 0.0
        if method != "gdi":
            self._open_dxgi(strict=method == "dxgi")

    def _open_dxgi(self, strict: bool = False) -> None:
        try:
            import dxcam

            factory = vars(dxcam)["__factory"]
            outputs = []
            for d, device_outputs in enumerate(factory.outputs):
                for o, output in enumerate(device_outputs):
                    camera = dxcam.create(device_idx=d, output_idx=o, output_color="BGR")
                    rect = output.desc.DesktopCoordinates  # refreshed by create()
                    outputs.append(_Output(camera, rect.left, rect.top, rect.right, rect.bottom))
            self._outputs = outputs
        except Exception:
            if strict:
                raise
            traceback.print_exc()
            self._outputs = []
            self._failed_at = time.monotonic()

    def _dxgi(self, left: int, top: int, width: int, height: int) -> np.ndarray | None:
        if self.method == "gdi":
            return None
        if not self._outputs:
            if self._outputs is not None and time.monotonic() - self._failed_at < _RETRY_DXGI_AFTER:
                return None
            self._open_dxgi()
        output = next((o for o in self._outputs or [] if o.contains(left, top, width, height)), None)
        if output is None:
            return None
        region = (left - output.left, top - output.top, left - output.left + width, top - output.top + height)
        try:
            frame = output.camera.grab(region=region, new_frame_only=True)
        except Exception:
            traceback.print_exc()
            self.close_dxgi()
            self._failed_at = time.monotonic()
            return None
        if frame is None:
            # Nothing on this monitor changed since the last grab.
            return output.frame if output.region == region else None
        output.region, output.frame = region, np.ascontiguousarray(frame)
        return output.frame

    def grab(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        frame = self._dxgi(left, top, width, height)
        if frame is not None:
            return frame.copy()
        shot = self._sct.grab({"left": left, "top": top, "width": width, "height": height})
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])  # BGRA -> BGR

    def close_dxgi(self) -> None:
        for output in self._outputs or []:
            try:
                output.camera.release()
            except Exception:
                pass
        self._outputs = []

    def close(self) -> None:
        self.close_dxgi()
        self._sct.close()
