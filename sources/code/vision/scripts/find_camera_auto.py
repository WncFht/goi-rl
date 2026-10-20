"""
Automatically find and read camera Y position from Getting Over It
using pymem's Mono support.

Usage: python scripts/find_camera_auto.py
  (Game must be running and in-game)
"""

import sys
import time
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pymem
import pymem.process


def find_camera_y() -> float | None:
    """Try multiple strategies to find the camera Y coordinate."""

    # Strategy 1: Search for a known nearby value (camera Z = -20)
    # In Vector3 {x, y, z}, z is always -20 for this game's orthographic camera
    print("[1/3] Attaching to GettingOverIt.exe...")
    try:
        pm = pymem.Pymem("GettingOverIt.exe")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Make sure Getting Over It is running and you're in-game.")
        return None

    print("  Attached.")

    # Strategy 2: Read from the save game in registry (always works, might be stale)
    print("[2/3] Reading camera Y from registry save data (fallback)...")
    import winreg
    import re
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Bennett Foddy\Getting Over It",
                             0, winreg.KEY_READ)
        for i in range(winreg.QueryInfoKey(key)[1]):
            name, data, _ = winreg.EnumValue(key, i)
            if name == "SaveGame0_h1867918426":
                match = re.search(rb"<camPos>.*?<y>([-\d.]+)</y>", data, re.DOTALL)
                if match:
                    cam_y = float(match.group(1))
                    print(f"  Registry camPos.y = {cam_y:.4f}")
        winreg.CloseKey(key)
    except Exception as e:
        print(f"  Registry read failed: {e}")

    # Strategy 3: Try to use Mono to find the class
    print("[3/3] Trying Mono-based class discovery...")
    try:
        mono_module = pymem.process.module_from_name(
            pm.process_handle, "mono-2.0-bdwgc.dll"
        )
        mono_base = mono_module.lpBaseOfDll
        print(f"  Mono module base: 0x{mono_base:X}")

        # Get the Mono root domain
        # In Unity Mono, root domain function is mono_get_root_domain
        # We need to resolve the function address
        root_domain_func_name = b"mono_get_root_domain"

        # Try to find the function in the mono module
        from ctypes import windll, c_void_p, c_char_p

        kernel32 = windll.kernel32
        proc_handle = pm.process_handle

        # Get the function address
        func_addr = kernel32.GetProcAddress(mono_base, c_char_p(root_domain_func_name))
        if func_addr:
            print(f"  mono_get_root_domain @ 0x{func_addr:X}")

            # Create remote thread to call the function - too complex for now
            # Let's try a simpler approach
        else:
            print("  Could not find mono_get_root_domain in mono module")

        # Alternative: try to find exported mono functions
        print("\n  Looking for Mono exported functions...")
        # Mono exports are in mono-2.0-bdwgc.dll
        exports = [
            "mono_get_root_domain",
            "mono_domain_get",
            "mono_thread_attach",
            "mono_assembly_loaded",
            "mono_class_from_name",
            "mono_object_get_class",
            "mono_class_get_field_from_name",
            "mono_field_get_value",
        ]
        for exp in exports:
            addr = kernel32.GetProcAddress(mono_base, c_char_p(exp.encode()))
            if addr:
                print(f"    ✓ {exp} @ 0x{addr:X}")
            else:
                print(f"    ✗ {exp} - not found")

    except Exception as e:
        print(f"  Mono discovery failed: {e}")

    print("\n=== Next Steps ===")
    print("  The registry fallback works. It gives camera Y from the last save.")
    print("  For real-time training, we need either:")
    print("  1. Find the memory address manually with Cheat Engine (searching")
    print("     for a more unique value like hingePos ~= -105000)")
    print("  2. Or use a BepInEx plugin to log camera Y to a file")
    print("  3. Or use the registry fallback (slower but functional)")
    print()
    print("  Recommendation: try searching for hingePos (~-105000)")
    print("  in Cheat Engine next. It's a much more unique value.")


if __name__ == "__main__":
    find_camera_y()
