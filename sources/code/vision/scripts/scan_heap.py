"""
Smart heap scanner: finds the camera position Vector3 by comparing
memory snapshots before and after the player moves.

Strategy:
1. Take a snapshot of all plausible (x, y, -20.0) Vector3s in heap memory
2. Have the user move in the game
3. Take another snapshot
4. Find the Vector3 whose Y changed proportionally to the movement

Usage: python scripts/scan_heap.py
"""

import sys
import time
import struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pymem


def scan_heap_for_minus20(pm: pymem.Pymem, max_mb: int = 500):
    """Quick scan of process heap for Vector3s with z = -20.0."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    proc = pm.process_handle

    MEM_COMMIT = 0x1000
    PAGE_READWRITE = 0x04

    candidates = {}  # addr_y -> (x, y)

    addr = 0
    scanned = 0
    SCAN_LIMIT = max_mb * 1024 * 1024

    minus_20 = struct.pack('<f', -20.0)

    print(f'[scan] Scanning heap (up to {max_mb}MB)...')

    while addr < 0x7FFFFFFFFFFF and scanned < SCAN_LIMIT:
        # Quick VirtualQuery
        mbi = ctypes.create_string_buffer(48)  # MEMORY_BASIC_INFORMATION size ~48 bytes
        ret = kernel32.VirtualQueryEx(
            ctypes.c_void_p(proc),
            ctypes.c_void_p(addr),
            mbi, 48
        )
        if ret == 0:
            break

        # Parse MBI manually (simpler than ctypes.Structure)
        base = struct.unpack_from('Q', mbi.raw, 0)[0]   # BaseAddress
        alloc_base = struct.unpack_from('Q', mbi.raw, 8)[0]
        region_size = struct.unpack_from('Q', mbi.raw, 24)[0]
        state = struct.unpack_from('I', mbi.raw, 32)[0]
        protect = struct.unpack_from('I', mbi.raw, 36)[0]

        if (state == MEM_COMMIT and
            region_size > 4096 and
            protect & 0xFF in (0x04, 0x02, 0x20, 0x40, 0x80)):  # readable

            # Cap region read size
            read_size = min(region_size, 2 * 1024 * 1024)
            try:
                chunk = pm.read_bytes(base, read_size)
            except Exception:
                addr = base + region_size
                continue

            # Search for -20.0f
            pos = 0
            while True:
                pos = chunk.find(minus_20, pos)
                if pos == -1:
                    break
                if pos >= 8:
                    try:
                        x = struct.unpack('<f', chunk[pos - 8:pos - 4])[0]
                        y = struct.unpack('<f', chunk[pos - 4:pos])[0]
                    except Exception:
                        pos += 1
                        continue
                    # Camera: x in [-10, 10], y in [-10, 500]
                    if -10 <= x <= 10 and -10 <= y <= 500:
                        addr_y = base + pos - 4  # address of y
                        candidates[addr_y] = (x, y)
                pos += 1

            scanned += read_size
            if scanned % (50 * 1024 * 1024) == 0:
                print(f'  {scanned // (1024*1024)}MB scanned, {len(candidates)} candidates...')

        addr = base + region_size

    print(f'[scan] Done: {scanned // (1024*1024)}MB scanned, {len(candidates)} candidates')
    return candidates


def read_snapshot(pm: pymem.Pymem, candidates: dict) -> dict:
    """Re-read Y values for all candidate addresses."""
    snapshot = {}
    for addr_y in candidates:
        try:
            snapshot[addr_y] = pm.read_float(addr_y)
        except Exception:
            snapshot[addr_y] = None
    return snapshot


def main():
    print('=' * 50)
    print('  Camera Auto-Finder (Heap Scanner)')
    print('=' * 50)
    print()
    print('This script will:')
    print('  1. Scan game memory for camera candidates')
    print('  2. Wait for you to CLIMB UP in the game')
    print('  3. Find the candidate whose Y increased')
    print()

    try:
        pm = pymem.Pymem('GettingOverIt.exe')
    except Exception:
        print('ERROR: GettingOverIt.exe not running. Start the game first.')
        return

    # Phase 1: Initial scan
    input('Press Enter to start the memory scan...')
    candidates = scan_heap_for_minus20(pm, max_mb=300)

    if not candidates:
        print('[!] No camera candidates found. Try climbing a bit then re-run.')
        pm.close_process()
        return

    print(f'\n[scan] Found {len(candidates)} candidates.')

    # Phase 2: Take initial snapshot
    print('[snap] Recording initial values...')
    snap1 = read_snapshot(pm, candidates)

    # Phase 3: Wait for movement
    print('\n========================================')
    print('  NOW GO CLIMB UP in the game!')
    print('  Move as far up as you can, then')
    print('  come back and press Enter.')
    print('========================================')
    input('\nPress Enter AFTER you have climbed up...')

    # Phase 4: Take second snapshot
    print('[snap] Recording new values...')
    snap2 = read_snapshot(pm, candidates)

    # Phase 5: Compare
    print('[cmp] Finding the camera...')
    changed = []
    for addr_y, y1 in snap1.items():
        y2 = snap2.get(addr_y)
        if y1 is not None and y2 is not None:
            delta = y2 - y1
            if delta > 0.1:  # must have gone UP (we asked user to climb)
                # Check: is the corresponding z still -20?
                try:
                    z = pm.read_float(addr_y + 4)  # z is right after y
                    if abs(z - (-20.0)) < 0.01:  # z must be -20.0
                        x = pm.read_float(addr_y - 4)  # x is right before y
                        changed.append((addr_y, x, y1, y2, delta, z))
                except Exception:
                    pass

    pm.close_process()

    if not changed:
        print('[!] No candidates went UP with z=-20.0.')
        print('    Try climbing further and re-run.')
        return

    # Sort by delta (largest change = most likely the camera)
    changed.sort(key=lambda x: -abs(x[4]))

    print(f'\n[result] Found {len(changed)} camera-like candidates:')
    for i, (addr, x, y1, y2, delta, z) in enumerate(changed[:5]):
        print(f'  #{i + 1}: addr_y = 0x{addr:X}')
        print(f'         x={x:.3f}, y: {y1:.3f} → {y2:.3f} (Δ={delta:+.3f}), z={z:.3f}')

    if changed:
        best = changed[0]
        print(f'\n=== RESULT ===')
        print(f'  Camera Y address: 0x{best[0]:X}')
        print(f'  Verify: x={best[1]:.3f}, y={best[2]:.3f}→{best[3]:.3f}')

        # Save to config
        import json
        from pathlib import Path
        config = {
            "camera_y_address": f"0x{best[0]:X}",
            "found_by": "heap_scan",
            "verified_y_change": best[4],
        }
        config_path = Path(__file__).parent.parent / "config" / "memory_addresses.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f'  Saved to config/memory_addresses.json')
        print()

        # Live test
        import pymem as pm2
        pm2 = pymem.Pymem('GettingOverIt.exe')
        print('[live] Reading camera Y (Ctrl+C to stop)...')
        last = best[2]
        try:
            while True:
                y = pm2.read_float(best[0])
                if abs(y - last) > 0.01:
                    print(f'  y = {y:.4f}  ({y - last:+.4f})')
                    last = y
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        pm2.close_process()


if __name__ == '__main__':
    main()
