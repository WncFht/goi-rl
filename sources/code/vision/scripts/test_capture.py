"""
Test script: verify screen capture from Getting Over It.
Run while the game is open (windowed mode).

Usage: python scripts/test_capture.py
"""

import sys
import time
import cv2

sys.path.insert(0, ".")

from src.game_interface.window import WindowManager
from src.game_interface.capture import ScreenCapture


def main():
    wm = WindowManager()
    hwnd = wm.find_window()
    if hwnd is None:
        print("[test_capture] ERROR: 'Getting Over It' window not found.")
        print("              Make sure the game is running in windowed mode.")
        return

    rect = wm.get_rect()
    print(f"[test_capture] Window found — HWND {hwnd}, rect {rect}")
    print(f"[test_capture] Client size: {wm.get_client_size()}")

    cap = ScreenCapture(wm, target_fps=60, grayscale=True, resize_to=(84, 84))

    # Benchmark
    print("[test_capture] Benchmarking capture FPS (300 frames)...")
    result = cap.benchmark(300)
    print(f"  → FPS: {result['fps']}, successful: {result['successful']}/{result['total']}")

    # Show a few captured frames
    print("[test_capture] Capturing 5 frames for visual check...")
    cap.stop()
    cap.start()
    for i in range(5):
        frame = cap.grab()
        if frame is not None:
            print(f"  Frame {i + 1}: shape={frame.shape}, dtype={frame.dtype}, "
                  f"min={frame.min():.0f}, max={frame.max():.0f}, mean={frame.mean():.1f}")

    # Save a sample image
    frame = cap.grab()
    if frame is not None:
        cv2.imwrite("data/sample_capture.png", frame)
        print("[test_capture] Saved sample image → data/sample_capture.png")

    cap.stop()
    print("[test_capture] Done.")


if __name__ == "__main__":
    main()
