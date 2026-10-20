"""
Game process management for Getting Over It.
Launch, kill, and restart the game process.
"""

import os
import subprocess
import time
from typing import Optional

import win32process
import win32con
import win32api
import win32gui
import psutil


# Common Steam install paths for Getting Over It
_COMMON_PATHS = [
    r"D:\SteamLibrary\steamapps\common\Getting Over It",
    r"C:\Program Files (x86)\Steam\steamapps\common\Getting Over It",
    r"D:\Steam\steamapps\common\Getting Over It",
    r"E:\Steam\steamapps\common\Getting Over It",
]

_EXE_NAME = "GettingOverIt.exe"
_PROCESS_NAME = "GettingOverIt.exe"


class GameLauncher:
    """Launch / kill / restart the Getting Over It process."""

    def __init__(self, exe_path: Optional[str] = None):
        """
        Args:
            exe_path: Full path to GettingOverIt.exe.
                      If None, auto-detect from common Steam paths.
        """
        self._exe_path = exe_path or self._find_exe()
        self._process: Optional[subprocess.Popen] = None

    # ---- path detection ----

    @staticmethod
    def _find_exe() -> Optional[str]:
        """Auto-detect GettingOverIt.exe from common Steam library locations."""
        for steam_dir in _COMMON_PATHS:
            candidate = os.path.join(steam_dir, _EXE_NAME)
            if os.path.isfile(candidate):
                return candidate
        return None

    @property
    def exe_path(self) -> Optional[str]:
        return self._exe_path

    # ---- find running process ----

    @staticmethod
    def find_process() -> Optional[int]:
        """Return PID of a running GettingOverIt.exe, or None."""
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.info["name"] == _PROCESS_NAME:
                    return proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    @staticmethod
    def is_running() -> bool:
        return GameLauncher.find_process() is not None

    # ---- launch ----

    def launch(self, windowed: bool = True, width: int = 800, height: int = 600) -> bool:
        """Launch the game.

        Args:
            windowed: Run in windowed mode (adds -popupwindow flag).
            width: Target window width.
            height: Target window height.
        """
        if self._exe_path is None:
            print("[launcher] Cannot find GettingOverIt.exe — set exe_path manually")
            return False

        if self.is_running():
            print("[launcher] Game is already running")
            return True

        args = [self._exe_path]
        if windowed:
            args.extend(["-screen-width", str(width), "-screen-height", str(height)])
            args.append("-popupwindow")

        try:
            self._process = subprocess.Popen(
                args,
                cwd=os.path.dirname(self._exe_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[launcher] Started {self._exe_path} (PID {self._process.pid})")
            return True
        except Exception as e:
            print(f"[launcher] Failed to launch: {e}")
            return False

    # ---- kill ----

    @staticmethod
    def kill() -> bool:
        """Kill all GettingOverIt.exe processes."""
        killed = False
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.info["name"] == _PROCESS_NAME:
                    proc.kill()
                    proc.wait(timeout=5)
                    killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
        if killed:
            print("[launcher] Killed GettingOverIt.exe")
        return killed

    # ---- restart ----

    def restart(self, **kwargs) -> bool:
        """Kill and relaunch the game."""
        self.kill()
        time.sleep(2)
        return self.launch(**kwargs)
