"""
Custom Stable-Baselines3 callbacks for Getting Over It training.
"""

import os
import threading
import time
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecEnv


class MetricsCallback(BaseCallback):
    """Logs custom training metrics to TensorBoard.

    Tracks:
      - Episode count and average length
      - Max camera Y height reached
      - Steps per second
      - Reward components breakdown (if available in infos)
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._log_dir = log_dir
        self._start_time = time.perf_counter()
        self._episode_rewards: list = []
        self._episode_lengths: list = []
        self._max_heights: list = []
        self._episode_count: int = 0
        self._last_log_time: float = time.perf_counter()

    def _on_step(self) -> bool:
        # Collect episode info from the "infos" array
        # In DummyVecEnv, infos is [{}]; in SubprocVecEnv, it's [{}, {}, ...]
        for info in self.locals.get("infos", []):
            if info is None:
                continue

            if "camera_y" in info:
                self._max_heights.append(info.get("episode_max_y", info["camera_y"]))

        # Check for episode ends via the dones tensor
        dones = self.locals.get("dones", None)
        if dones is not None:
            for i, done in enumerate(dones):
                if done:
                    self._episode_count += 1
                    # Log reward from the rewards buffer
                    if "rewards" in self.locals:
                        rewards = self.locals["rewards"]  # VecEnv gives per-env rewards
                        if i < len(rewards):
                            self._episode_rewards.append(rewards[i])

        # Log to TensorBoard every ~30 seconds
        now = time.perf_counter()
        if now - self._last_log_time >= 30:
            self._log_metrics()
            self._last_log_time = now

        return True

    def _on_rollout_end(self) -> None:
        """Called after each PPO rollout."""
        self._log_metrics()

    def _log_metrics(self):
        """Write accumulated metrics to the logger."""
        total_steps = self.num_timesteps
        elapsed = time.perf_counter() - self._start_time
        fps = total_steps / max(elapsed, 1)

        # Episode metrics
        if self._episode_rewards:
            self.logger.record("metrics/episode_reward_mean", np.mean(self._episode_rewards[-50:]))
            self.logger.record("metrics/episode_reward_max", max(self._episode_rewards[-50:]))
        if self._max_heights:
            self.logger.record("metrics/camera_y_max", max(self._max_heights[-100:]))

        self.logger.record("metrics/episodes", self._episode_count)
        self.logger.record("metrics/steps_per_second", fps)
        self.logger.record("metrics/total_hours", elapsed / 3600.0)

        # Dump the record so it appears in TensorBoard
        self.logger.dump(step=total_steps)


class CrashRecoveryCallback(BaseCallback):
    """Detects game/environment crashes and triggers safe shutdown.

    If the game process dies, this callback sets a flag so the training
    loop can exit gracefully and be resumed from the last checkpoint.
    """

    def __init__(self, env: VecEnv, verbose: int = 0):
        super().__init__(verbose)
        self._env = env
        self._consecutive_errors: int = 0
        self.max_consecutive_errors: int = 50
        self.crashed: bool = False

    def _on_step(self) -> bool:
        # Check for error info in infos
        for info in self.locals.get("infos", []):
            if isinstance(info, dict) and "error" in info:
                self._consecutive_errors += 1
            else:
                self._consecutive_errors = 0

        if self._consecutive_errors >= self.max_consecutive_errors:
            print(f"[CrashRecovery] {self._consecutive_errors} consecutive errors "
                  f"— game may have crashed")
            self.crashed = True
            return False  # Stop training

        return True


class StopFlagCallback(BaseCallback):
    """Stops training gracefully when an external flag is set.

    Use this with a global hotkey thread: press a key → set the event →
    training exits on the next step with full checkpoint save.
    """

    def __init__(self, stop_event: threading.Event, label: str = "", verbose: int = 0):
        super().__init__(verbose)
        self._stop_event = stop_event
        self._label = label or "stop flag"

    def _on_step(self) -> bool:
        if self._stop_event.is_set():
            if self.verbose >= 1:
                print(f"[StopFlagCallback] {self._label} — stopping training")
            return False
        return True


def create_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    save_freq: int = 100_000,
    verbose: int = 0,
) -> list:
    """Create the standard set of callbacks for training.

    Args:
        checkpoint_dir: Directory to save model checkpoints.
        log_dir: Directory for TensorBoard logs.
        save_freq: Save a checkpoint every N steps.
        verbose: Verbosity level.

    Returns:
        List of callbacks.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="goi_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    metrics = MetricsCallback(log_dir=log_dir, verbose=verbose)

    return [checkpoint, metrics]
