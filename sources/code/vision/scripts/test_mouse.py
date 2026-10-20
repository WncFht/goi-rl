"""
Test script: verify mouse input injection works with Getting Over It.
The game reads raw mouse delta — this script sends small movements.

Usage: python scripts/test_mouse.py
  (Make sure Getting Over It is running and in-game)
"""

import sys
import time

sys.path.insert(0, ".")

from src.game_interface.mouse import MouseController


def main():
    print("[test_mouse] === Mouse Input Test ===")
    print("  This will send small mouse movements.")
    print("  Make sure Getting Over It is running and you're in the game world.")
    print("  You have 3 seconds to switch to the game window...")
    time.sleep(3)

    mc = MouseController(max_delta=200)

    # Test small movements in each direction
    print("[test_mouse] Small move RIGHT...")
    mc.move_relative(50, 0)
    time.sleep(0.5)

    print("[test_mouse] Small move LEFT...")
    mc.move_relative(-50, 0)
    time.sleep(0.5)

    print("[test_mouse] Small move UP...")
    mc.move_relative(0, 50)
    time.sleep(0.5)

    print("[test_mouse] Small move DOWN...")
    mc.move_relative(0, -50)
    time.sleep(0.5)

    # Test apply_action (normalised interface)
    print("[test_mouse] Testing apply_action with random actions...")
    import random

    for i in range(10):
        dx = random.uniform(-1, 1)
        dy = random.uniform(-1, 1)
        mc.apply_action(dx, dy)
        time.sleep(0.1)

    print("[test_mouse] Done. Check if the hammer moved in-game.")


if __name__ == "__main__":
    main()
