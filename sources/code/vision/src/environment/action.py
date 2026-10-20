"""
Action processing for Getting Over It RL environment.
Handles denormalisation of agent actions to actual mouse deltas.
"""

import numpy as np
from typing import Optional


class ActionProcessor:
    """Denormalises RL agent actions to mouse deltas."""

    def __init__(
        self,
        max_delta: float = 200.0,
        action_clip: float = 1.0,
    ):
        """
        Args:
            max_delta: Maximum pixel delta per step.
                       Agent action in [-1, 1] maps to [-max_delta, max_delta].
            action_clip: Clip normalised actions to [-action_clip, action_clip].
                         Default 1.0 (no extra clipping beyond tanh output).
        """
        self.max_delta = max_delta
        self.action_clip = action_clip

    def denormalise(self, action: np.ndarray) -> np.ndarray:
        """Convert normalised action to pixel deltas.

        Args:
            action: Normalised action [dx, dy] in [-1, 1].

        Returns:
            Pixel delta array [dx, dy] as float32.
        """
        clipped = np.clip(action, -self.action_clip, self.action_clip)
        return (clipped * self.max_delta).astype(np.float32)

    def denormalise_smooth(
        self,
        action: np.ndarray,
        prev_action: Optional[np.ndarray] = None,
        smoothing: float = 0.3,
    ) -> np.ndarray:
        """Denormalise with temporal smoothing to reduce jitter.

        Args:
            action: Current normalised action.
            prev_action: Previous denormalised action (pixel space).
            smoothing: EMA smoothing factor (0 = no smoothing, 1 = full smoothing).

        Returns:
            Smoothed pixel delta array.
        """
        raw = self.denormalise(action)
        if prev_action is not None:
            return prev_action * smoothing + raw * (1 - smoothing)
        return raw
