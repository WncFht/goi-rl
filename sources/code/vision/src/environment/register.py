"""
Register the Getting Over It environment with Gymnasium.
"""

import gymnasium as gym

from .goi_env import GettingOverItEnv


def register_env(env_id: str = "GettingOverIt-v0"):
    """Register the environment so it can be created with gym.make()."""
    # Unregister if already registered (e.g., re-import in a notebook)
    if env_id in gym.registry:
        del gym.registry[env_id]

    gym.register(
        id=env_id,
        entry_point="src.environment.goi_env:GettingOverItEnv",
        max_episode_steps=10_000,
    )
