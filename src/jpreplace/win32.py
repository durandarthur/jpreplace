"""Small Win32 helpers (ctypes): capture exclusion, window rects, global
hotkeys, and finding / following other applications' windows."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")

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

_ENUM_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [_ENUM_PROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
for _name in ("IsWindow", "IsWindowVisible", "IsIconic"):
    getattr(user32, _name).argtypes = [wintypes.HWND]
    getattr(user32, _name).restype = wintypes.BOOL
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
user32.SetWindowPos.restype = wintypes.BOOL
user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.MonitorFromWindow.restype = wintypes.HMONITOR
dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long


class _MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo)]
user32.GetMonitorInfoW.restype = wintypes.BOOL
user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetWindow.restype = wintypes.HWND
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
user32.GetCursorPos.restype = wintypes.BOOL

_LRESULT = ctypes.c_ssize_t
_LL_MOUSE_PROC = ctypes.WINFUNCTYPE(_LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, _LL_MOUSE_PROC, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = _LRESULT
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL


class _MouseHookInfo(ctypes.Structure):
    """MSLLHOOKSTRUCT"""

    _fields_ = [
        ("pt", wintypes.POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x20
_WS_EX_TOOLWINDOW = 0x80
_WS_EX_APPWINDOW = 0x40000
_GW_OWNER = 4
_SW_RESTORE = 9
_WH_MOUSE_LL = 14
_WM_LBUTTONDOWN, _WM_LBUTTONUP, _WM_RBUTTONDOWN, _WM_RBUTTONUP = 0x201, 0x202, 0x204, 0x205
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_DWMWA_CLOAKED = 14
_GA_ROOT = 2
_HWND_TOPMOST = wintypes.HWND(-1)
_SWP_NOSIZE, _SWP_NOMOVE, _SWP_NOZORDER, _SWP_NOACTIVATE = 0x1, 0x2, 0x4, 0x10
_MONITOR_DEFAULTTONEAREST = 2
# The desktop and the taskbar: never something to bind to.
_SHELL_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}

Rect = tuple[int, int, int, int]  # left, top, width, height


class _JobLimits(ctypes.Structure):
    """JOBOBJECT_EXTENDED_LIMIT_INFORMATION, flattened."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
        ("IoInfo", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_job_handles: list[int] = []  # kept open for the life of this process


def kill_with_parent(process_handle: int) -> bool:
    """Tie a child process to ours: when this process ends for any reason
    (including a crash or Task Manager), Windows terminates the child too."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return False
    limits = _JobLimits()
    limits.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
    ok = ok and kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process_handle))
    if ok:
        _job_handles.append(job)
    return bool(ok)


def exclude_from_capture(hwnd: int) -> bool:
    """Hide a window from screen capture (it stays visible on the monitor)."""
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Window rectangle (left, top, width, height) in physical screen pixels."""
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def client_rect(hwnd: int) -> Rect:
    """A window's client area (its content, without title bar and borders)
    in physical screen pixels."""
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, rect.right, rect.bottom


def monitor_rect(hwnd: int, work_area: bool = False) -> Rect:
    """The monitor (physical pixels) that a window is mostly on: all of it,
    or the part not taken by the taskbar."""
    info = _MonitorInfo()
    info.cbSize = ctypes.sizeof(info)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONEAREST), ctypes.byref(info))
    r = info.rcWork if work_area else info.rcMonitor
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def move_window(hwnd: int, rect: Rect) -> None:
    """Place a window at a rectangle in physical pixels."""
    user32.SetWindowPos(hwnd, None, *rect, _SWP_NOZORDER | _SWP_NOACTIVATE)


def raise_topmost(hwnd: int) -> None:
    """Put an always-on-top window back above other always-on-top windows
    (full-screen games often make themselves topmost)."""
    user32.SetWindowPos(hwnd, _HWND_TOPMOST, 0, 0, 0, 0, _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE)


def is_window(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd))


def is_cloaked(hwnd: int) -> bool:
    """Hidden by the compositor: on another virtual desktop, or a suspended UWP app."""
    cloaked = wintypes.DWORD(0)
    dwmapi.DwmGetWindowAttribute(hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
    return bool(cloaked.value)


def is_on_screen(hwnd: int) -> bool:
    """Exists, is shown, not minimised and not cloaked."""
    return is_window(hwnd) and user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and not is_cloaked(hwnd)


def window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def window_title(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def _class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(128)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


def foreground_window() -> int:
    return user32.GetForegroundWindow() or 0


def is_shell(hwnd: int) -> bool:
    return _class_name(hwnd) in _SHELL_CLASSES


def overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def window_at(x: int, y: int, exclude: set[int] | None = None) -> int | None:
    """The topmost application window under a screen point (physical
    pixels), skipping click-through overlays, the desktop, the taskbar and
    this process's own windows (or, if given, the windows in `exclude`)."""
    own = os.getpid()
    found: list[int] = []

    def visit(hwnd: int, _param: int) -> bool:
        ours = hwnd in exclude if exclude is not None else window_pid(hwnd) == own
        if ours or not is_on_screen(hwnd) or is_shell(hwnd):
            return True
        if user32.GetWindowLongW(hwnd, _GWL_EXSTYLE) & _WS_EX_TRANSPARENT:
            return True
        try:
            left, top, width, height = window_rect(hwnd)
        except OSError:
            return True
        if width > 1 and height > 1 and left <= x < left + width and top <= y < top + height:
            found.append(hwnd)
            return False  # EnumWindows goes from the top of the Z-order down
        return True

    user32.EnumWindows(_ENUM_PROC(visit), 0)
    return user32.GetAncestor(found[0], _GA_ROOT) if found else None


def cursor_pos() -> tuple[int, int]:
    point = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def bring_to_front(hwnd: int) -> None:
    """Restore a window if it is minimised, and give it the focus."""
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, _SW_RESTORE)
    user32.SetForegroundWindow(hwnd)


def process_name(pid: int) -> str:
    """The executable's file name, e.g. "notepad.exe" ("" if not allowed)."""
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buffer))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return os.path.basename(buffer.value)
    finally:
        kernel32.CloseHandle(handle)


def list_windows() -> list[tuple[int, str, str]]:
    """(hwnd, title, program) of the windows one can switch to, as in
    Alt+Tab, front to back; minimised ones included."""
    own = os.getpid()
    found: list[tuple[int, str, str]] = []

    def visit(hwnd: int, _param: int) -> bool:
        if not user32.IsWindowVisible(hwnd) or is_cloaked(hwnd) or is_shell(hwnd):
            return True
        pid = window_pid(hwnd)
        title = window_title(hwnd)
        style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        app_window = style & _WS_EX_APPWINDOW or not (style & _WS_EX_TOOLWINDOW or user32.GetWindow(hwnd, _GW_OWNER))
        if pid != own and title and app_window:
            found.append((hwnd, title, process_name(pid)))
        return True

    user32.EnumWindows(_ENUM_PROC(visit), 0)
    return found


class ClickCatcher:
    """Catches the next mouse click anywhere on screen, before any
    application sees it (a low-level mouse hook). `on_click(button, x, y)`
    is called on release with "left" or "right" and physical screen
    coordinates. Must be created on a thread that runs a message loop (the
    Qt GUI thread), and removed with `remove()`."""

    def __init__(self, on_click) -> None:
        self.on_click = on_click
        self._proc = _LL_MOUSE_PROC(self._callback)  # keep a reference: Windows calls it
        self._hook = user32.SetWindowsHookExW(_WH_MOUSE_LL, self._proc, kernel32.GetModuleHandleW(None), 0)
        if not self._hook:
            raise ctypes.WinError(ctypes.get_last_error())

    def _callback(self, code: int, message: int, info: int) -> int:
        if code == 0 and message in (_WM_LBUTTONDOWN, _WM_LBUTTONUP, _WM_RBUTTONDOWN, _WM_RBUTTONUP):
            if message in (_WM_LBUTTONUP, _WM_RBUTTONUP):
                point = _MouseHookInfo.from_address(info).pt
                self.on_click("left" if message == _WM_LBUTTONUP else "right", point.x, point.y)
            return 1  # swallowed: the application under the cursor never sees it
        return user32.CallNextHookEx(None, code, message, info)

    def remove(self) -> None:
        if self._hook:
            user32.UnhookWindowsHookEx(self._hook)
            self._hook = None


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
