"""
Replay a saved episode from training (frame by frame).

Usage:
    python -m src.analysis.replay_episode --model checkpoints/goi_ppo_1000000_steps.zip
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.goi_env import GettingOverItEnv


def replay_episode(
    model_path: str,
    deterministic: bool = True,
    record: bool = False,
    output_path: str = "data/replay.mp4",
    fps: int = 10,
):
    """Run one episode and replay it with OpenCV display.

    Args:
        model_path: Path to trained model.
        deterministic: Use deterministic actions.
        record: Save frames as an MP4 video.
        output_path: Output video path.
        fps: Recording FPS.
    """
    print(f"[replay] Loading {model_path}")
    model = PPO.load(model_path)

    env = GettingOverItEnv(
        step_rate=10.0,
        max_episode_steps=10_000,
    )

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    max_y = info.get("camera_y", 0)

    frames = []

    print("[replay] Running episode...")
    print("  Press 'q' to stop, 's' to toggle stochastic/deterministic")

    is_deterministic = deterministic

    while not done:
        action, _ = model.predict(obs, deterministic=is_deterministic)
        obs, reward, done, info = env.step(action)

        total_reward += reward
        if info.get("camera_y", 0) > max_y:
            max_y = info["camera_y"]

        # Get the raw frame for display
        raw_frame = env._capture._last_frame
        if raw_frame is not None:
            # Overlay info
            display = raw_frame.copy()
            if display.ndim == 2:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

            cv2.putText(display, f"Max Y: {max_y:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Reward: {total_reward:.1f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Current Y: {info.get('camera_y', 0):.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Mode: {'det' if is_deterministic else 'stoch'}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if record:
                frames.append(display)

            cv2.imshow("GOI Replay", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                is_deterministic = not is_deterministic
                print(f"  Switched to {'deterministic' if is_deterministic else 'stochastic'}")

        if info.get("termination_reason") == "fall":
            print(f"  [Fall!]  Max Y: {max_y:.2f}")

    env.close()
    cv2.destroyAllWindows()

    print(f"\n[replay] Episode finished:")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Max camera Y: {max_y:.2f}")

    # Save video
    if record and frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        print(f"  Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay a GOI-RL agent episode")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint .zip")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions")
    parser.add_argument("--record", action="store_true",
                        help="Record frames to MP4")
    parser.add_argument("--output", type=str, default="data/replay.mp4",
                        help="Output video path")
    args = parser.parse_args()

    replay_episode(
        model_path=args.model,
        deterministic=not args.stochastic,
        record=args.record,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
