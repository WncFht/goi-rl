"""
Mouse input injection via Win32 SendInput.
Sends relative mouse movements that Unity's Input.GetAxis("Mouse X/Y") can read.

Cursor locking: during training, the system cursor is hidden and clipped to a
small rectangle in the bottom-right corner of the primary monitor.  This keeps
the cursor from flying around your desktop while still allowing SendInput to
deliver relative mouse deltas to the game.
"""

import atexit
import ctypes
import time
from ctypes import wintypes

# ---- Win32 types & constants ----

user32 = ctypes.windll.user32

# SendInput constants
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

# System metrics
SM_CXSCREEN = 0
SM_CYSCREEN = 1


class RECT(ctypes.Structure):
    _fields_ = [
        ("left",   wintypes.LONG),
        ("top",    wintypes.LONG),
        ("right",  wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("mi", MOUSEINPUT),
    ]


class MouseController:
    """Inject relative mouse movements into the system input stream.

    Cursor locking keeps the physical cursor out of your way during training
    without interfering with the game's ability to read mouse deltas.
    """

    def __init__(self, max_delta: int = 200, lock_cursor: bool = True):
        """
        Args:
            max_delta: Maximum absolute pixel delta per move call.
                       Raw agent actions will be scaled to [-max_delta, max_delta].
            lock_cursor: If True, hide and clip the system cursor so it doesnʼt
                         fly around the desktop during training.  The game still
                         receives mouse input normally.
        """
        self.max_delta = max_delta
        self._lock_cursor = lock_cursor
        self._dx_accum = 0.0
        self._dy_accum = 0.0
        self._cursor_locked = False
        self._show_cursor_count = 0

    # ---- cursor locking ----

    def lock(self) -> None:
        """Hide the cursor and clip it to a rectangle in the bottom-right
        corner of the primary monitor.  Safe to call multiple times — only the
        first call has an effect."""
        if self._cursor_locked or not self._lock_cursor:
            return

        screen_w = user32.GetSystemMetrics(SM_CXSCREEN)
        screen_h = user32.GetSystemMetrics(SM_CYSCREEN)

        # Rect large enough for ±max_delta movement in any direction (plus margin)
        rect_size = 2 * self.max_delta + 100

        self._clip_rect = RECT()
        self._clip_rect.left   = max(0, screen_w - rect_size)
        self._clip_rect.top    = max(0, screen_h - rect_size)
        self._clip_rect.right  = screen_w
        self._clip_rect.bottom = screen_h

        # Park the cursor at the centre of the clip rect
        ctypes.windll.user32.SetCursorPos(
            screen_w - rect_size // 2,
            screen_h - rect_size // 2,
        )

        # Confine it
        ctypes.windll.user32.ClipCursor(ctypes.byref(self._clip_rect))

        # Hide it (ShowCursor returns the new display count; hide until negative)
        while ctypes.windll.user32.ShowCursor(False) >= 0:
            self._show_cursor_count += 1

        self._cursor_locked = True

        # Best-effort restore on normal Python exit
        atexit.register(self._atexit_unlock)

    def unlock(self) -> None:
        """Release the cursor clip and make it visible again."""
        if not self._cursor_locked:
            return

        # Remove clipping
        ctypes.windll.user32.ClipCursor(None)

        # Restore visibility (reverse the ref-count decrements from lock)
        for _ in range(self._show_cursor_count):
            ctypes.windll.user32.ShowCursor(True)

        self._cursor_locked = False
        self._show_cursor_count = 0

    def _atexit_unlock(self) -> None:
        """Called at process exit as a safety net."""
        if self._cursor_locked:
            ctypes.windll.user32.ClipCursor(None)
            # ShowCursor(True) enough times to get back to visible
            for _ in range(self._show_cursor_count + 10):
                ctypes.windll.user32.ShowCursor(True)

    # ---- raw SendInput wrapper ----

    @staticmethod
    def move_relative(dx: int, dy: int) -> bool:
        """Send a single relative mouse movement via SendInput.

        Unity reads these deltas as Input.GetAxis("Mouse X/Y").
        The dx/dy values are in raw screen pixels (or arbitrary units —
        Unity normalises them based on system mouse settings).
        """
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi.dx = dx
        inp.mi.dy = dy
        inp.mi.mouseData = 0
        inp.mi.dwFlags = MOUSEEVENTF_MOVE  # relative movement (no ABSOLUTE flag)
        inp.mi.time = 0
        inp.mi.dwExtraInfo = None

        result = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        return result > 0

    # ---- agent-friendly interface ----

    def apply_action(self, dx_norm: float, dy_norm: float) -> bool:
        """Apply a normalised action from the RL agent.

        Locks the cursor on first call (if lock_cursor=True was passed).

        Args:
            dx_norm: Normalised dx in [-1, 1]
            dy_norm: Normalised dy in [-1, 1]

        Returns:
            True if the input was sent successfully.
        """
        # Lock cursor on first use
        if self._lock_cursor and not self._cursor_locked:
            self.lock()

        # Scale from [-1, 1] to pixel deltas
        dx = int(dx_norm * self.max_delta)
        dy = int(dy_norm * self.max_delta)

        # Accumulate fractional parts for sub-pixel precision over time
        dx_frac = dx_norm * self.max_delta - dx
        dy_frac = dy_norm * self.max_delta - dy
        self._dx_accum += dx_frac
        self._dy_accum += dy_frac

        if abs(self._dx_accum) >= 1.0:
            dx += int(self._dx_accum)
            self._dx_accum -= int(self._dx_accum)
        if abs(self._dy_accum) >= 1.0:
            dy += int(self._dy_accum)
            self._dy_accum -= int(self._dy_accum)

        return self.move_relative(dx, dy)

    # ---- click helpers (for menus / restarting) ----

    @staticmethod
    def click() -> bool:
        """Send a left mouse click (down + up)."""
        inp_down = INPUT()
        inp_down.type = INPUT_MOUSE
        inp_down.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
        inp_down.mi.time = 0
        inp_down.mi.dwExtraInfo = None

        inp_up = INPUT()
        inp_up.type = INPUT_MOUSE
        inp_up.mi.dwFlags = MOUSEEVENTF_LEFTUP
        inp_up.mi.time = 0
        inp_up.mi.dwExtraInfo = None

        r1 = user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(inp_down))
        time.sleep(0.02)
        r2 = user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(inp_up))
        return r1 > 0 and r2 > 0
