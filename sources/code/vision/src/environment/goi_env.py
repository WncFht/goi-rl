"""
Gymnasium environment for Getting Over It RL training.

Wraps the game interface (screen capture + mouse control) and state reader
into a standard Gymnasium Env with visual observations and continuous actions.

Architecture:
  1. Capture game window → raw frame
  2. Read camera Y from memory/registry → reward signal
  3. Process frame → stacked observation
  4. Denormalise action → mouse delta → SendInput
  5. Compute reward → step return
"""

import time
from typing import Optional, Tuple, Any

import gymnasium as gym
import numpy as np

from src.game_interface.window import WindowManager
from src.game_interface.capture import ScreenCapture
from src.game_interface.mouse import MouseController
from src.state_reader.camera_reader import CameraReader
from .observation import ObservationProcessor
from .reward import RewardCalculator
from .action import ActionProcessor


class GettingOverItEnv(gym.Env):
    """Gymnasium environment for Getting Over It.

    Observation: 4 stacked grayscale frames (4, 84, 84), uint8
    Action: Continuous [dx, dy] in [-1, 1], float32
    """

    metadata = {"render_modes": []}

    # ---- init ----

    def __init__(
        self,
        step_rate: float = 10.0,          # Hz
        max_episode_steps: int = 10_000,  # ~16 min at 10 Hz
        frame_stack: int = 4,
        obs_size: Tuple[int, int] = (84, 84),
        max_mouse_delta: float = 200.0,
        height_scale: float = 10.0,
        fall_threshold: float = 3.0,
        death_penalty: float = 5.0,        # Penalty for black-screen death (reduced)
        black_threshold: float = 5.0,       # Mean pixel value below this = black screen
        black_frames_required: int = 2,     # Consecutive black frames to confirm death
        death_grace_steps: int = 30,        # No death penalty for first N steps of episode
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Timing
        self._step_rate = step_rate
        self._step_interval = 1.0 / step_rate
        self._max_episode_steps = max_episode_steps

        # Death detection
        self._death_penalty = death_penalty
        self._black_threshold = black_threshold
        self._black_frames_required = black_frames_required
        self._death_grace_steps = death_grace_steps
        self._consecutive_black: int = 0

        # Components
        self._window = WindowManager()
        self._capture = ScreenCapture(
            self._window,
            target_fps=int(step_rate * 2),  # capture faster than we step
            grayscale=True,
            resize_to=obs_size,
        )
        self._mouse = MouseController(max_delta=int(max_mouse_delta))
        self._camera = CameraReader()

        # Processors
        self._obs_proc = ObservationProcessor(
            frame_stack=frame_stack,
            grayscale=True,
            resize_to=obs_size,
        )
        self._reward_calc = RewardCalculator(
            height_scale=height_scale,
            fall_threshold=fall_threshold,
        )
        self._action_proc = ActionProcessor(max_delta=max_mouse_delta)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=self._obs_proc.shape,
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # State
        self._step_count: int = 0
        self._episode_count: int = 0
        self._total_steps: int = 0

    # ---- Gymnasium API ----

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment for a new episode.

        The game handles respawn natively on death.
        We capture the initial frame and height after respawn.
        """
        super().reset(seed=seed)

        self._step_count = 0
        self._episode_count += 1

        # Ensure capture is running
        self._capture.stop()
        if not self._capture.start():
            raise RuntimeError(
                "Cannot start screen capture. Is the game running in windowed mode?"
            )

        # Read initial height
        init_y = self._camera.y
        self._reward_calc.reset(init_y)
        self._consecutive_black = 0

        # Capture initial frame and fill buffer
        initial_frame = self._wait_for_frame()
        self._obs_proc.reset(initial_frame)

        obs = self._obs_proc._get_obs()
        info = {"camera_y": init_y, "episode": self._episode_count}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step.

        Args:
            action: Normalised [dx, dy] in [-1, 1].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        self._total_steps += 1

        # Denormalise and inject mouse action
        action = np.asarray(action, dtype=np.float32).flatten()
        self._mouse.apply_action(action[0], action[1])

        # Wait for the step interval
        time.sleep(self._step_interval)

        # Capture next frame
        frame = self._capture.grab()
        if frame is None:
            # Return zero reward, no termination
            obs = self._obs_proc._get_obs()
            return obs, 0.0, False, False, {"error": "capture failed"}

        # Detect black screen = death (fell into water / void)
        # Require consecutive black frames to avoid false positives
        frame_mean = float(np.mean(frame))
        if frame_mean < self._black_threshold:
            self._consecutive_black += 1
        else:
            self._consecutive_black = 0
        is_black = self._consecutive_black >= self._black_frames_required

        # Read camera Y
        current_y = self._camera.y

        # Compute reward
        reward, reward_components, is_fall = self._reward_calc.compute(
            current_y, action
        )

        # Black screen death: penalty + terminate, but only after grace period
        grace_active = self._step_count <= self._death_grace_steps
        if is_black and not grace_active:
            reward -= self._death_penalty
            reward_components["reward/death"] = -self._death_penalty
        elif is_black and grace_active:
            reward_components["reward/death"] = 0.0  # grace: no penalty, just log

        # Build observation
        obs = self._obs_proc.push(frame)

        # Termination conditions
        # During grace period, black screen does NOT terminate (agent can explore)
        terminated = is_fall or (is_black and not grace_active)
        truncated = self._step_count >= self._max_episode_steps

        # Info dict
        info = {
            "camera_y": current_y,
            "episode_max_y": self._reward_calc.episode_max_y,
            "episode": self._episode_count,
            "step": self._step_count,
            "total_steps": self._total_steps,
            **reward_components,
        }

        if terminated:
            if is_black:
                info["termination_reason"] = "death"
            elif is_fall:
                info["termination_reason"] = "fall"
        if is_black and grace_active:
            info["death_grace"] = True  # black screen ignored during grace period

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources."""
        self._capture.stop()
        self._mouse.unlock()
        # CameraReader uses file I/O — no cleanup needed

    # ---- helpers ----

    def _wait_for_frame(self, max_attempts: int = 30) -> np.ndarray:
        """Wait for a valid frame from the capture device."""
        for _ in range(max_attempts):
            frame = self._capture.grab()
            if frame is not None:
                return frame
            time.sleep(0.05)

        # Last resort: return a blank frame
        h, w = self._obs_proc.resize_to
        return np.zeros((h, w), dtype=np.uint8)
