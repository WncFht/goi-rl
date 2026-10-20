"""
Observation processing for Getting Over It RL environment.
Handles frame stacking, resizing, and observation construction.
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple


class ObservationProcessor:
    """Preprocesses raw frames into RL observations with frame stacking."""

    def __init__(
        self,
        frame_stack: int = 4,
        grayscale: bool = True,
        resize_to: Tuple[int, int] = (84, 84),
    ):
        """
        Args:
            frame_stack: Number of consecutive frames to stack.
                         Provides temporal information (velocity, motion).
            grayscale: Whether to convert frames to single-channel grayscale.
            resize_to: (width, height) to resize frames to.
        """
        self.frame_stack = frame_stack
        self.grayscale = grayscale
        self.resize_to = resize_to
        self._buffer: deque = deque(maxlen=frame_stack)

    # ---- observation shape ----

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of a single observation."""
        channels = self.frame_stack
        h, w = self.resize_to
        # Channel-first format for PyTorch / Stable-Baselines3
        return (channels, h, w)

    @property
    def dtype(self):
        return np.uint8

    # ---- frame processing ----

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a raw frame into the format for the stack.

        Args:
            frame: Raw frame from DXCam (BGR or grayscale numpy array).

        Returns:
            Processed frame at the target size.
        """
        # Convert to grayscale if needed
        if self.grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize
        if self.resize_to is not None:
            frame = cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_AREA)

        return frame.astype(np.uint8)

    # ---- buffer management ----

    def reset(self, initial_frame: np.ndarray):
        """Reset the frame buffer with the initial frame duplicated."""
        processed = self.process(initial_frame)
        self._buffer.clear()
        for _ in range(self.frame_stack):
            self._buffer.append(processed)

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Push a new frame and return the stacked observation.

        Args:
            frame: The latest raw frame.

        Returns:
            Stacked observation array, shape (frame_stack, H, W).
        """
        processed = self.process(frame)
        self._buffer.append(processed)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Build the stacked observation from the buffer."""
        # Stack along channel axis → (H, W, C)
        stacked = np.stack(list(self._buffer), axis=0)
        return stacked.astype(np.uint8)

    @property
    def is_ready(self) -> bool:
        """Whether the buffer has enough frames for a valid observation."""
        return len(self._buffer) == self.frame_stack
