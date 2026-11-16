"""Small Win32 helpers (ctypes): capture exclusion, window rects, global hotkeys."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

WDA_EXCLUDEFROMCAPTURE = 0x11
WM_HOTKEY = 0x0312
_MODIFIERS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
_MOD_NOREPEAT = 0x4000
_KEYS = {
    "space": 0x20, "enter": 0x0D, "esc": 0x1B, "escape": 0x1B, "tab": 0x09,
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
}

user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
user32.RegisterHotKey.restype = wintypes.BOOL
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
user32.UnregisterHotKey.restype = wintypes.BOOL


def exclude_from_capture(hwnd: int) -> bool:
    """Hide a window from screen capture (it stays visible on the monitor)."""
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Window rectangle (left, top, width, height) in physical screen pixels."""
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'ctrl+alt+t' -> (modifier flags, virtual key code)."""
    mods, vk = 0, 0
    for part in spec.lower().replace(" ", "").split("+"):
        if part in _MODIFIERS:
            mods |= _MODIFIERS[part]
        elif part in _KEYS:
            vk = _KEYS[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        else:
            raise ValueError(f"unknown key {part!r} in hotkey {spec!r}")
    if not vk:
        raise ValueError(f"hotkey {spec!r} has no key")
    return mods, vk


def register_hotkey(hwnd: int, hotkey_id: int, spec: str) -> bool:
    mods, vk = parse_hotkey(spec)
    return bool(user32.RegisterHotKey(hwnd, hotkey_id, mods | _MOD_NOREPEAT, vk))


def unregister_hotkey(hwnd: int, hotkey_id: int) -> None:
    user32.UnregisterHotKey(hwnd, hotkey_id)


def hotkey_id_from_message(message: int) -> int | None:
    """Given a pointer to a native MSG, return the hotkey id if it is WM_HOTKEY."""
    msg = wintypes.MSG.from_address(int(message))
    return int(msg.wParam) if msg.message == WM_HOTKEY else None
