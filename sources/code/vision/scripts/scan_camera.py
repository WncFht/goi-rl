"""
Scan game memory for the camera Transform position.

Strategy: camera.z is ALWAYS -20.0 (orthographic camera at fixed distance).
Find all -20.0 floats, check adjacent bytes for plausible x/y values,
track which one moves when the player climbs/falls.

Usage: python scripts/scan_camera.py
"""

import sys
import time
import struct
import ctypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pymem
import pymem.process
from pymem.ptypes import RemotePointer


def scan_for_camera(pm: pymem.Pymem):
    """Find the camera Transform position Vector3 in game memory."""

    # -20.0 as IEEE 754 float in little-endian bytes
    minus_20_bytes = struct.pack("<f", -20.0)

    print("[scan] Scanning memory for camera Z = -20.0...")
    print("  This may take ~30 seconds...")

    # We'll scan the main game module and the mono heap area
    # Get the game module and mono module ranges
    game_module = pymem.process.module_from_name(pm.process_handle, "GettingOverIt.exe")
    mono_module = pymem.process.module_from_name(pm.process_handle, "mono-2.0-bdwgc.dll")

    candidates = []
    chunk_size = 4096
    modules_to_scan = [game_module, mono_module]

    for mod in modules_to_scan:
        base = mod.lpBaseOfDll
        size = mod.SizeOfImage
        print(f"  Scanning {mod.name}: base=0x{base:X}, size={size:,} bytes")

        for offset in range(0, size, chunk_size):
            try:
                chunk = pm.read_bytes(base + offset, min(chunk_size, size - offset))
            except Exception:
                continue

            # Search for -20.0f pattern
            pos = 0
            while True:
                pos = chunk.find(minus_20_bytes, pos)
                if pos == -1:
                    break

                # We found a -20.0. Now check the 8 bytes BEFORE it
                # which would be the x and y floats if this is position.z
                if pos >= 8:
                    try:
                        x = struct.unpack("<f", chunk[pos - 8:pos - 4])[0]
                        y = struct.unpack("<f", chunk[pos - 4:pos])[0]
                    except Exception:
                        pos += 1
                        continue

                    # Filter: camera position has plausible x (-5 to 5) and y (-5 to 300)
                    if -5 <= x <= 5 and -5 <= y <= 300:
                        abs_addr = base + offset + pos  # address of the z value
                        candidates.append({
                            "addr_x": abs_addr - 8,  # x starts 8 bytes before z
                            "addr_y": abs_addr - 4,  # y starts 4 bytes before z
                            "addr_z": abs_addr,
                            "x": x,
                            "y": y,
                            "z": -20.0,
                        })

                pos += 1

    print(f"\n[scan] Found {len(candidates)} candidates with plausible x,y values")
    return candidates


def track_changes(pm: pymem.Pymem, candidates: list):
    """Monitor which candidate changes when the player moves."""
    if not candidates:
        print("[scan] No candidates to track.")
        return None

    print("\n[track] Now tracking changes...")
    print("  Go into the game and CLIMB UP or FALL DOWN.")
    print("  Then come back and press Enter here.")
    print("  I'll check which candidate's y value changed the most.")
    input("\n  Press Enter when you've moved in the game...")

    print("\n[track] Reading current values...")
    changed = []
    for i, c in enumerate(candidates):
        try:
            new_y = pm.read_float(c["addr_y"])
            delta = abs(new_y - c["y"])
            if delta > 0.01:
                changed.append((i, c, new_y, delta))
        except Exception:
            pass

    if not changed:
        print("  No candidates changed. Try moving more (climb further).")
        return None

    print(f"\n[track] {len(changed)} candidates changed:")
    changed.sort(key=lambda x: -x[3])  # sort by delta descending
    for rank, (i, c, new_y, delta) in enumerate(changed[:10]):
        print(f"  #{rank + 1}: y {c['y']:.3f} → {new_y:.3f} "
              f"(Δ={delta:.3f}), addr_y=0x{c['addr_y']:X}")

    if changed:
        best = changed[0]
        print(f"\n  🎯 BEST CANDIDATE: addr_y = 0x{best[1]['addr_y']:X}")
        return best[1]

    return None


def main():
    print("[scan] === Camera Auto-Scanner ===")
    print()

    try:
        pm = pymem.Pymem("GettingOverIt.exe")
    except Exception:
        print("ERROR: GettingOverIt.exe not found. Is the game running?")
        return

    candidates = scan_for_camera(pm)

    if len(candidates) > 100:
        print(f"\n  Too many candidates ({len(candidates)}). "
              "Let's filter by tracking movement...")

    result = track_changes(pm, candidates)

    if result:
        print(f"\n[scan] === SUCCESS ===")
        print(f"  Camera Y address: 0x{result['addr_y']:X}")
        print(f"  Add this to config/memory_addresses.json")

        # Quick test
        print("\n[scan] Testing live reads (CTRL+C to stop)...")
        try:
            last = result["y"]
            while True:
                y = pm.read_float(result["addr_y"])
                if y != last:
                    print(f"  y = {y:.4f}  (changed by {y - last:+.4f})")
                    last = y
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    pm.close_process()


if __name__ == "__main__":
    main()
