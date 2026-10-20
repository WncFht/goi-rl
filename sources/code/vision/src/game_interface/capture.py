"""
Screen capture for Getting Over It via DXCam (DXGI Desktop Duplication).
Provides fast GPU-based screen capture of the game window region.
"""

import time
from typing import Optional, Tuple

import numpy as np
import cv2

from .window import WindowManager


class ScreenCapture:
    """Captures the game window using DXCam with optional downscaling."""

    def __init__(
        self,
        window: WindowManager,
        target_fps: int = 60,
        grayscale: bool = True,
        resize_to: Optional[Tuple[int, int]] = (84, 84),
    ):
        """
        Args:
            window: WindowManager instance targeting the game.
            target_fps: Desired capture frame rate for DXCam's internal loop.
            grayscale: Convert captured frame to single-channel grayscale.
            resize_to: (width, height) to resize the captured image to,
                       or None to keep native size.
        """
        self._window = window
        self._target_fps = target_fps
        self._grayscale = grayscale
        self._resize_to = resize_to
        self._camera = None
        self._last_frame: Optional[np.ndarray] = None

    # ---- lifecycle ----

    def start(self) -> bool:
        """Initialise the DXCam camera. Call once before grab()."""
        import dxcam

        rect = self._window.get_rect()
        if rect is None:
            print("[capture] Game window not found - cannot start DXCam")
            return False

        left, top, right, bottom = rect
        region = (left, top, right, bottom)

        self._camera = dxcam.create(
            region=region,
            output_color="BGR",  # DXCam produces BGR natively
        )

        # Optionally start a continuous capture at target FPS.
        # We use single-frame grab() instead for simplicity.
        print(f"[capture] DXCam started - region {region}, target {self._target_fps} FPS")
        return True

    def stop(self):
        """Release the DXCam camera."""
        if self._camera is not None:
            try:
                del self._camera
            except Exception:
                pass
            self._camera = None

    # ---- capture ----

    def grab(self) -> Optional[np.ndarray]:
        """Capture a single frame from the game window.

        Returns:
            numpy array (H, W) if grayscale, (H, W, 3) if BGR, or None on failure.
        """
        if self._camera is None:
            if not self.start():
                return None

        if self._window.is_minimized():
            # Window is minimised — DXCam may return stale data.
            # Return the last valid frame if available.
            return self._last_frame

        try:
            frame = self._camera.grab()
        except Exception:
            # DXCam may raise if the GPU device is lost
            frame = None

        if frame is None:
            # No new frame yet; return last known good frame
            return self._last_frame

        # DXCam returns a numpy array in BGR (or BGRA) format
        frame = self._preprocess(frame)
        self._last_frame = frame
        return frame

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert raw DXCam frame to the desired format."""
        # DXCam can return BGRA (4 channels); strip alpha if present
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        if self._grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._resize_to is not None:
            frame = cv2.resize(frame, self._resize_to, interpolation=cv2.INTER_AREA)

        return frame

    # ---- benchmarking ----

    def benchmark(self, num_frames: int = 300) -> dict:
        """Measure actual capture FPS over *num_frames* frames."""
        self.stop()
        if not self.start():
            return {"fps": 0, "successful": 0, "total": num_frames}

        successful = 0
        start = time.perf_counter()
        for _ in range(num_frames):
            frame = self.grab()
            if frame is not None:
                successful += 1
        elapsed = time.perf_counter() - start

        fps = successful / elapsed if elapsed > 0 else 0
        return {"fps": round(fps, 1), "successful": successful, "total": num_frames}
