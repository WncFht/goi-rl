"""
Plot training metrics from TensorBoard event files.

Usage:
    python -m src.analysis.plot_training --logdir logs/
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("[plot] tensorboard not installed; install with: pip install tensorboard")
    raise


def load_tensorboard_data(logdir: str) -> dict:
    """Load scalar data from TensorBoard event files."""
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"[plot] No event files found in {logdir}")
        return {}

    # Use the most recent event file
    event_file = max(event_files, key=os.path.getmtime)
    print(f"[plot] Loading {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
        }
    return data


def print_summary(data: dict):
    """Print a summary of training progress."""
    if not data:
        print("[plot] No data to summarise")
        return

    print("\n=== Training Summary ===\n")

    # Reward
    if "rollout/ep_rew_mean" in data:
        d = data["rollout/ep_rew_mean"]
        print(f"  Mean episode reward:")
        print(f"    Start: {d['values'][0]:.2f}")
        print(f"    End:   {d['values'][-1]:.2f}")
        print(f"    Max:   {max(d['values']):.2f}")

    # Custom metrics
    for tag in sorted(data.keys()):
        if tag.startswith("metrics/"):
            d = data[tag]
            print(f"  {tag}:")
            print(f"    Latest: {d['values'][-1]:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Plot GOI-RL training metrics")
    parser.add_argument("--logdir", type=str, default="logs/",
                        help="Path to TensorBoard log directory")
    args = parser.parse_args()

    data = load_tensorboard_data(args.logdir)
    print_summary(data)

    print("\n[plot] For full visualisation, run: tensorboard --logdir logs/")


if __name__ == "__main__":
    main()
