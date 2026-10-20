"""
Read camera position from the BepInEx CameraLogger plugin output file.

The plugin writes camera position (x,y,z) to camera_y.txt at 20Hz.
This reader polls the file for the latest value.
"""

import os
import time
from typing import Optional, Tuple


class CameraReader:
    """Reads camera Y from the BepInEx plugin output file."""

    def __init__(self, file_path: Optional[str] = None):
        """
        Args:
            file_path: Path to camera_y.txt. Defaults to game directory.
        """
        if file_path is None:
            file_path = os.path.join(
                os.environ.get("GOI_DIR", r"D:\SteamLibrary\steamapps\common\Getting Over It"),
                "camera_y.txt",
            )
        self._file_path = file_path
        self._last_y: float = 0.0
        self._last_x: float = 0.0
        self._last_z: float = -20.0
        self._last_read_time: float = 0.0
        self._last_file_mtime: float = 0.0

    # ---- main read method ----

    def read(self) -> Optional[Tuple[float, float, float]]:
        """Read the current camera position.

        Returns:
            (x, y, z) tuple, or None if the file cannot be read.
        """
        try:
            # Check if file was modified since last read (avoids unnecessary I/O)
            mtime = os.path.getmtime(self._file_path)
            if mtime == self._last_file_mtime:
                return (self._last_x, self._last_y, self._last_z)

            self._last_file_mtime = mtime

            with open(self._file_path, "r") as f:
                line = f.read().strip()

            parts = line.split(",")
            if len(parts) == 3:
                self._last_x = float(parts[0])
                self._last_y = float(parts[1])
                self._last_z = float(parts[2])
                self._last_read_time = time.perf_counter()
                return (self._last_x, self._last_y, self._last_z)

        except (FileNotFoundError, ValueError, IOError):
            pass

        return None

    @property
    def y(self) -> float:
        """Get the last known camera Y (read optimistically, returns cached value on failure)."""
        result = self.read()
        if result is not None:
            return result[1]
        return self._last_y

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def is_available(self) -> bool:
        return os.path.exists(self._file_path)
