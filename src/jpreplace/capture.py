"""Screen capture of a rectangle given in physical pixels."""

from __future__ import annotations

import mss
import numpy as np


class ScreenGrabber:
    """Must be created and used on the same thread."""

    def __init__(self) -> None:
        self._sct = mss.mss()

    def grab(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        shot = self._sct.grab({"left": left, "top": top, "width": width, "height": height})
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])  # BGRA -> BGR

    def close(self) -> None:
        self._sct.close()
