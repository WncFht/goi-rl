"""
Monitor camera Y height from registry in real-time.
Run this, then play the game — see how fast the registry updates.

Usage: python scripts/monitor_height.py
"""

import sys
import time
import re
import winreg
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def read_cam_y_from_registry() -> float | None:
    """Read camPos.y from the most recent registry save."""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Bennett Foddy\Getting Over It",
            0, winreg.KEY_READ,
        )
        best = None
        for i in range(winreg.QueryInfoKey(key)[1]):
            name, data, _ = winreg.EnumValue(key, i)
            if name.startswith("SaveGame"):
                match = re.search(rb"<camPos>.*?<y>([-\d.]+)</y>", data, re.DOTALL)
                if match:
                    best = float(match.group(1))
        winreg.CloseKey(key)
        return best
    except Exception:
        return None


def main():
    print("[monitor] Polling camera Y from registry every 0.5 seconds...")
    print("  Go play the game! Climb, fall, move around.")
    print("  Values below show the last saved camPos.y")
    print("  Watch how often it changes.")
    print()
    print(f"{'Time':>8s}  {'camPos.y':>10s}  {'Status'}")
    print("-" * 40)

    last_y = None
    changes = 0
    start = time.perf_counter()

    try:
        while True:
            y = read_cam_y_from_registry()
            elapsed = time.perf_counter() - start
            status = ""

            if y is None:
                status = "READ ERROR"
            elif last_y is None:
                status = "INIT"
            elif y != last_y:
                status = "🔼 CHANGED!"
                changes += 1
            else:
                status = "  (same)"

            print(f"{elapsed:7.1f}s  {y:>10.4f}  {status}")

            last_y = y
            time.sleep(0.5)

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - start
        print(f"\n[monitor] Done. {changes} changes in {elapsed:.1f}s "
              f"({elapsed / max(changes, 1):.1f}s per change)")


if __name__ == "__main__":
    main()
