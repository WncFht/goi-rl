"""
Memory reader for Getting Over It camera Y coordinate.
Uses pymem to read the camera position from the game's Mono heap.

Supports two modes:
1. Direct offset: known static offset from the game module base
2. Pointer chain: multi-level pointer dereferencing
3. Registry fallback: read camPos.y from the registry save data (slower, less frequent)

The exact address/offsets must be found using Cheat Engine's Mono Dissect feature.
See the accompanying guide for step-by-step instructions.
"""

import json
import os
import time
from typing import Optional, Tuple

import pymem
import pymem.process
import winreg


class MemoryReader:
    """Reads the camera Y coordinate from Getting Over It's memory."""

    PROCESS_NAME = "GettingOverIt.exe"
    MODULE_NAME = "mono-2.0-bdwgc.dll"  # Mono runtime module

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to memory_addresses.json.
                         Defaults to config/memory_addresses.json relative to project root.
        """
        self._pm: Optional[pymem.Pymem] = None
        self._base_address: int = 0
        self._offsets: list[int] = []
        self._use_registry_fallback: bool = False

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config", "memory_addresses.json",
            )
        self._config_path = config_path
        self._load_config()

        self._last_y: float = 0.0
        self._last_read_time: float = 0.0

    # ---- config ----

    def _load_config(self):
        """Load stored memory offsets from config file."""
        if not os.path.exists(self._config_path):
            self._use_registry_fallback = True
            return

        try:
            with open(self._config_path, "r") as f:
                cfg = json.load(f)
            self._offsets = cfg.get("pointer_offsets", [])
            if self._offsets:
                print(f"[memory] Loaded pointer chain: {self._offsets}")
                self._use_registry_fallback = False
            else:
                self._use_registry_fallback = True
        except Exception as e:
            print(f"[memory] Failed to load config: {e}")
            self._use_registry_fallback = True

    # ---- pymem lifecycle ----

    def attach(self) -> bool:
        """Attach to the Getting Over It process."""
        try:
            self._pm = pymem.Pymem(self.PROCESS_NAME)
            self._base_address = pymem.process.module_from_name(
                self._pm.process_handle, self.MODULE_NAME
            ).lpBaseOfDll
            print(f"[memory] Attached to {self.PROCESS_NAME}, "
                  f"mono base = 0x{self._base_address:X}")
            return True
        except pymem.exception.ProcessNotFound:
            print(f"[memory] Process '{self.PROCESS_NAME}' not found")
            return False
        except pymem.exception.ModuleNotFound:
            print(f"[memory] Module '{self.MODULE_NAME}' not found")
            return False
        except Exception as e:
            print(f"[memory] Attach failed: {e}")
            return False

    def detach(self):
        """Detach from the process."""
        if self._pm is not None:
            try:
                self._pm.close_process()
            except Exception:
                pass
            self._pm = None

    @property
    def is_attached(self) -> bool:
        return self._pm is not None

    # ---- memory reading ----

    def read_camera_y(self) -> Optional[float]:
        """Read the current camera Y coordinate from game memory.

        Returns:
            Float Y value, or None if the read failed.
        """
        if self._use_registry_fallback or not self._offsets:
            return self._read_from_registry()

        if self._pm is None:
            if not self.attach():
                return self._read_from_registry()

        try:
            # Starting address: mono module base + first offset
            addr = self._base_address + self._offsets[0]

            # Follow remaining pointer offsets
            for offset in self._offsets[1:]:
                addr = self._pm.read_int(addr) + offset

            value = self._pm.read_float(addr)
            self._last_y = value
            self._last_read_time = time.perf_counter()
            return value

        except (pymem.exception.MemoryReadError, pymem.exception.WinAPIError) as e:
            print(f"[memory] Read error: {e}, falling back to registry")
            self._use_registry_fallback = True
            return self._read_from_registry()

    def read_camera_y_robust(self) -> float:
        """Read camera Y with a local cache for robustness.

        If the read fails, returns the last known good value.
        Always returns a float (never None).
        """
        value = self.read_camera_y()
        if value is not None:
            return value
        return self._last_y

    # ---- registry fallback ----

    REGISTRY_KEY = r"Software\Bennett Foddy\Getting Over It"

    def _read_from_registry(self) -> Optional[float]:
        """Fallback: read camPos.y from the registry save data.

        The game periodically writes save state to the registry.
        This is not real-time but provides a rough height estimate.
        """
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, self.REGISTRY_KEY, 0, winreg.KEY_READ
            )
            # Try SaveGame0 first (most recent save)
            for save_name in ["SaveGame0", "SaveGame1"]:
                for i in range(winreg.QueryInfoKey(key)[1]):
                    name, data, _ = winreg.EnumValue(key, i)
                    if name.startswith(save_name):
                        value = self._parse_cam_y(data)
                        if value is not None:
                            winreg.CloseKey(key)
                            self._last_y = value
                            self._last_read_time = time.perf_counter()
                            return value
            winreg.CloseKey(key)
        except Exception:
            pass
        return self._last_y

    @staticmethod
    def _parse_cam_y(data: bytes) -> Optional[float]:
        """Extract camPos.y from raw save data XML."""
        import re
        match = re.search(rb"<camPos>.*?<y>([-\d.]+)</y>", data, re.DOTALL)
        if match:
            return float(match.group(1))
        return None

    # ---- address validation ----

    def validate_address(self, expected_range: Tuple[float, float] = (-100, 500)) -> bool:
        """Check that the read value is in a plausible range for camera Y."""
        value = self.read_camera_y()
        if value is None:
            return False
        return expected_range[0] <= value <= expected_range[1]
