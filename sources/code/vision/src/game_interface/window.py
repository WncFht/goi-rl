"""
Window management for Getting Over It.
Find, focus, resize the game window via Win32 API.
"""

import ctypes
import time
from typing import Optional, Tuple

import win32gui
import win32con
import win32process


class WindowManager:
    """Manages the Getting Over It game window."""

    WINDOW_TITLE = "Getting Over It"

    def __init__(self):
        self._hwnd: Optional[int] = None

    # ---- find / attach ----

    def find_window(self) -> Optional[int]:
        """Find the game window by title. Returns hwnd or None."""
        hwnd = win32gui.FindWindow(None, self.WINDOW_TITLE)
        if hwnd:
            self._hwnd = hwnd
        return hwnd

    @property
    def hwnd(self) -> Optional[int]:
        if self._hwnd and win32gui.IsWindow(self._hwnd):
            return self._hwnd
        return self.find_window()

    # ---- rect ----

    def get_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Return (left, top, right, bottom) client-area coordinates.
        Uses GetClientRect + ClientToScreen so the region matches the
        render surface regardless of window chrome.
        """
        hwnd = self.hwnd
        if hwnd is None:
            return None

        # Get client rect in client coordinates
        client_rect = win32gui.GetClientRect(hwnd)
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        right = left + client_rect[2]
        bottom = top + client_rect[3]
        return (left, top, right, bottom)

    def get_client_size(self) -> Optional[Tuple[int, int]]:
        """Return (width, height) of the client area."""
        hwnd = self.hwnd
        if hwnd is None:
            return None
        rect = win32gui.GetClientRect(hwnd)
        return (rect[2], rect[3])

    # ---- state checks ----

    def is_alive(self) -> bool:
        """Check if the game window still exists."""
        return self.hwnd is not None

    def is_minimized(self) -> bool:
        """Check if the window is minimized (IsIconic)."""
        hwnd = self.hwnd
        if hwnd is None:
            return False
        return bool(win32gui.IsIconic(hwnd))

    # ---- window manipulation ----

    def bring_to_foreground(self) -> bool:
        """Bring the game window to the foreground."""
        hwnd = self.hwnd
        if hwnd is None:
            return False
        try:
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.05)
            return True
        except Exception:
            return False

    def resize(self, width: int, height: int) -> bool:
        """Resize the game window (includes non-client area)."""
        hwnd = self.hwnd
        if hwnd is None:
            return False
        try:
            win32gui.SetWindowPos(
                hwnd, None, 0, 0, width, height,
                win32con.SWP_NOMOVE | win32con.SWP_NOZORDER,
            )
            time.sleep(0.1)
            return True
        except Exception:
            return False

    def move_to(self, x: int, y: int) -> bool:
        """Move the game window to a screen position."""
        hwnd = self.hwnd
        if hwnd is None:
            return False
        try:
            win32gui.SetWindowPos(
                hwnd, None, x, y, 0, 0,
                win32con.SWP_NOSIZE | win32con.SWP_NOZORDER,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def get_process_id(hwnd: int) -> int:
        """Get the process ID for a window handle."""
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return pid
