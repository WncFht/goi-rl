"""
Evaluate a trained Getting Over It RL agent.

Runs the agent with deterministic actions and records performance metrics.
Useful for checking progress without exploration noise.

Usage:
    python -m src.training.evaluate --model checkpoints/goi_ppo_1000000_steps.zip --episodes 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.goi_env import GettingOverItEnv
from src.environment.register import register_env
from stable_baselines3.common.vec_env import DummyVecEnv


def evaluate(
    model_path: str,
    num_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
) -> dict:
    """Evaluate a trained model.

    Args:
        model_path: Path to the model checkpoint (.zip).
        num_episodes: Number of episodes to run.
        deterministic: Use deterministic actions (no exploration).
        render: Whether to show captured frames (for debugging).

    Returns:
        Dict of evaluation metrics.
    """
    print(f"[evaluate] Loading model from {model_path}")
    model = PPO.load(model_path)

    def make_env():
        return GettingOverItEnv(
            step_rate=10.0,
            max_episode_steps=10_000,
        )

    env = DummyVecEnv([make_env])

    episode_rewards = []
    episode_lengths = []
    max_heights = []
    num_falls = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_max_y = -float("inf")
        ep_falls = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)

            ep_reward += reward[0]
            ep_steps += 1

            if info[0].get("camera_y", 0) > ep_max_y:
                ep_max_y = info[0]["camera_y"]

            if info[0].get("termination_reason") == "fall":
                ep_falls += 1

            if render:
                frame = env.envs[0]._capture._last_frame
                if frame is not None:
                    import cv2
                    cv2.imshow("Evaluation", frame)
                    cv2.waitKey(1)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        max_heights.append(ep_max_y)
        num_falls.append(ep_falls)

        print(f"  Episode {ep + 1}/{num_episodes}: "
              f"reward={ep_reward:.1f}, steps={ep_steps}, "
              f"max_y={ep_max_y:.2f}, falls={ep_falls}")

    env.close()
    if render:
        cv2.destroyAllWindows()

    results = {
        "episodes": num_episodes,
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "reward_min": min(episode_rewards),
        "reward_max": max(episode_rewards),
        "steps_mean": np.mean(episode_lengths),
        "max_height_mean": np.mean(max_heights),
        "max_height_max": max(max_heights),
        "falls_mean": np.mean(num_falls),
    }

    print(f"\n[evaluate] === Results ({num_episodes} episodes) ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GOI-RL agent")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint .zip")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions (with exploration noise)")
    parser.add_argument("--render", action="store_true",
                        help="Show captured frames during evaluation")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
    )


if __name__ == "__main__":
    main()
