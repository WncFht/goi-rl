"""
Main training script for Getting Over It RL agent.

Trains a PPO agent with CNN policy on the Getting Over It environment.
Includes crash recovery, TensorBoard auto-launch, and a global stop hotkey.

Usage:
    python -m src.training.train
    python -m src.training.train --resume checkpoints/goi_ppo_1000000_steps.zip

Stop training:
    Ctrl+Shift+Q   - global hotkey (works even when the game has focus)
    Ctrl+C         - classic terminal interrupt (needs terminal focus)
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

import keyboard
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.goi_env import GettingOverItEnv
from src.training.callbacks import create_callbacks, CrashRecoveryCallback, StopFlagCallback


PROJECT_ROOT = Path(__file__).parent.parent.parent

# Global events used by hotkey threads to signal the main thread
_stop_requested = threading.Event()
_start_ready = threading.Event()


# ---- Monkey-patch SB3 ep_info_buffer bug ----
# When episodes terminate very fast, SB3 sometimes puts bare ints into
# ep_info_buffer instead of dicts.  This causes len(int) -> TypeError.
# Patch _dump_logs to filter out non-dict entries gracefully.
def _patch_sb3_ep_info_buffer():
    from stable_baselines3.common import on_policy_algorithm

    _orig_dump_logs = on_policy_algorithm.OnPolicyAlgorithm._dump_logs

    def _safe_dump_logs(self, iteration: int) -> None:
        buf = self.ep_info_buffer
        # Filter out malformed entries (int instead of dict)
        clean = [e for e in buf if isinstance(e, dict) and len(e) > 0]
        if clean:
            self.ep_info_buffer.clear()
            self.ep_info_buffer.extend(clean)
        else:
            # All entries are bad — clear and skip this dump
            buf.clear()
            return
        _orig_dump_logs(self, iteration)

    on_policy_algorithm.OnPolicyAlgorithm._dump_logs = _safe_dump_logs

_patch_sb3_ep_info_buffer()


def load_config(config_path: str = None) -> dict:
    """Load training hyperparameters from YAML config."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "training_config.yaml"

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"[train] Loaded config from {config_path}")
        return cfg

    # Default config
    return {
        "algorithm": "PPO",
        "policy": "CnnPolicy",
        "learning_rate": 3.0e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": 10_000_000,
        "save_freq": 100_000,
    }


def make_env() -> GettingOverItEnv:
    """Create a new environment instance."""
    return GettingOverItEnv(
        step_rate=10.0,
        max_episode_steps=10_000,
        frame_stack=4,
        obs_size=(84, 84),
        max_mouse_delta=200.0,
        height_scale=10.0,
        fall_threshold=3.0,
    )


# ---- TensorBoard launcher ----

def _launch_tensorboard(log_dir: str) -> subprocess.Popen | None:
    """Start TensorBoard in the background and open the browser.

    Returns the Popen handle so the caller can terminate it on exit.
    """
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir,
             "--host", "localhost", "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        webbrowser.open("http://localhost:6006")
        print(f"[train] [TB] TensorBoard launched -> http://localhost:6006")
        return proc
    except Exception as e:
        print(f"[train] [!] TensorBoard auto-launch failed: {e}")
        print(f"[train]   Start manually: tensorboard --logdir {log_dir}")
        return None


# ---- global stop hotkey ----

def _register_stop_hotkey(hotkey: str = "ctrl+shift+q") -> None:
    """Register a global hotkey that signals the main thread to stop training.

    Runs in a daemon thread so it doesn't block the main process.
    The hotkey works even when the game window has focus.
    """
    _stop_requested.clear()

    def _on_hotkey() -> None:
        print(f"\n[train] [STOP] Hotkey ({hotkey}) pressed - stopping gracefully...")
        _stop_requested.set()

    keyboard.add_hotkey(hotkey, _on_hotkey)
    print(f"[train] [HOTKEY] Stop registered: {hotkey} (works globally)")


# ---- start hotkey (waits before training begins) ----

def _register_start_hotkey(hotkey: str = "ctrl+shift+s") -> None:
    """Register a global hotkey that unblocks training start."""
    _start_ready.clear()

    def _on_start() -> None:
        _start_ready.set()

    keyboard.add_hotkey(hotkey, _on_start)
    print(f"[train] [HOTKEY] Press {hotkey} to START training")


def _wait_for_start() -> None:
    """Block until the start hotkey is pressed."""
    print("[train] Waiting for start signal...")
    print()
    _start_ready.wait()  # blocks until hotkey sets the event
    print("[train] [START] Starting training now!\n")


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on Getting Over It")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint .zip file to resume from",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use: 'cuda', 'cpu', or 'auto'",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[train] Using device: {device}")

    # Paths
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    log_dir = PROJECT_ROOT / "logs"

    # Create environment
    print("[train] Creating environment...")
    env = DummyVecEnv([make_env])

    # Callbacks
    callbacks = create_callbacks(
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        save_freq=cfg.get("save_freq", 100_000),
    )
    crash_cb = CrashRecoveryCallback(env)
    callbacks.append(crash_cb)
    stop_cb = StopFlagCallback(_stop_requested, label="Ctrl+Shift+Q")
    callbacks.append(stop_cb)

    # Create or resume model
    total = args.timesteps or cfg.get("total_timesteps", 10_000_000)

    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = PPO.load(
            args.resume,
            env=env,
            device=device,
            tensorboard_log=str(log_dir),
        )
    else:
        print("[train] Creating new PPO model...")
        model = PPO(
            policy=cfg.get("policy", "CnnPolicy"),
            env=env,
            learning_rate=cfg.get("learning_rate", 3e-4),
            n_steps=cfg.get("n_steps", 2048),
            batch_size=cfg.get("batch_size", 64),
            n_epochs=cfg.get("n_epochs", 10),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            clip_range=cfg.get("clip_range", 0.2),
            ent_coef=cfg.get("ent_coef", 0.01),
            vf_coef=cfg.get("vf_coef", 0.5),
            max_grad_norm=cfg.get("max_grad_norm", 0.5),
            tensorboard_log=str(log_dir),
            device=device,
            verbose=1,
        )

    # Prepare
    print(f"[train] Training for {total:,} timesteps")
    print(f"[train] Checkpoints -> {checkpoint_dir}")
    print()

    # Auto-launch TensorBoard
    tb_proc = _launch_tensorboard(str(log_dir))

    # Register global stop hotkey (always active once registered)
    _register_stop_hotkey()

    print()

    # Register start hotkey and wait
    _register_start_hotkey()
    _wait_for_start()

    print(f"[train] [!] Cursor locked to bottom-right - it won't fly around.")
    print()

    start_time = time.perf_counter()

    try:
        model.learn(
            total_timesteps=total,
            callback=callbacks,
            reset_num_timesteps=(args.resume is None),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[train] Training interrupted by user")
    except Exception as e:
        print(f"\n[train] Training error: {e}")
        traceback.print_exc()

        # Save emergency checkpoint
        emergency_path = str(checkpoint_dir / "goi_ppo_emergency.zip")
        model.save(emergency_path)
        print(f"[train] Emergency checkpoint saved to {emergency_path}")
        return 1
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"[train] Training stopped after {elapsed / 3600:.1f} hours")

        # Shut down TensorBoard
        if tb_proc is not None:
            tb_proc.terminate()
            tb_proc.wait(timeout=5)
            print("[train] TensorBoard stopped.")

    # Save final model
    final_path = str(checkpoint_dir / "goi_ppo_final.zip")
    model.save(final_path)
    print(f"[train] Final model saved to {final_path}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
