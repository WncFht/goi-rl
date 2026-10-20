"""
Test script: verify memory reading of camera Y coordinate.
Prints the current camera height and checks that the read value is plausible.

Usage: python scripts/test_memory.py
  (Make sure Getting Over It is running and in-game)
"""

import sys
import time

sys.path.insert(0, ".")

from src.state_reader.memory_reader import MemoryReader


def main():
    print("[test_memory] === Memory Reader Test ===")
    print("  Checking registry fallback first (no Cheat Engine config needed)...")
    print()

    reader = MemoryReader()

    # Try reading (will use registry fallback since no offsets configured)
    for i in range(5):
        y = reader.read_camera_y()
        if y is not None:
            print(f"[test_memory] Read {i + 1}: camera_y = {y:.4f}  "
                  f"(source: {'registry' if reader._use_registry_fallback else 'memory'})")
        else:
            print(f"[test_memory] Read {i + 1}: FAILED")
        time.sleep(0.5)

    print()
    print("[test_memory] ---")
    print("  If values are showing (even if same), the registry fallback works.")
    print("  For real-time values, use Cheat Engine to find the memory address,")
    print("  then save offsets to config/memory_addresses.json.")
    print()
    print("  Cheat Engine guide: python -m src.state_reader.address_finder")


if __name__ == "__main__":
    main()
