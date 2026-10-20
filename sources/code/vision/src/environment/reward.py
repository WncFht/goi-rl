"""
Reward function for Getting Over It RL agent.

Reference: 茹莺 (gitee.com/Heuriz/goi-rl).  Key insight — asymmetric reward:
reward upward movement heavily, penalise downward movement lightly.
This mirrors the reference's "hammer alignment + upward velocity" reward
but adapted for camera-Y-only observations.

Design principles (from reference):
  1. Progress: asymmetric — reward up, barely penalise down (going down
     is sometimes necessary to swing upward).
  2. Upward bonus: extra reward for ANY upward movement — the densest signal
     we can provide without physics data.
  3. Fall penalty: proportional to how far you fell (not a flat -5).
  4. Action cost: small L2 penalty to discourage random flailing.
  5. Stuck penalty: if no progress for many consecutive steps, nudge the
     agent to try something different.
"""

from typing import Optional, Tuple
import numpy as np


class RewardCalculator:
    """Computes per-step reward from camera Y changes and action properties.

    The reward is deliberately ASYMMETRIC — heavily reward going up,
    barely penalise going down.  This is because:
      - The agent discovers upward movement by accident at first; we need
        that accident to register as a strong positive event.
      - Going down is part of the game's swinging mechanic and shouldn't
        be punished too harshly (the reference clips at -0.2 for this reason).
    """

    def __init__(
        self,
        height_scale: float = 10.0,            # Scale factor for height delta
        fall_threshold: float = 3.0,            # Min Y drop from peak to trigger fall
        fall_penalty_scale: float = 2.0,        # Multiplier for proportional fall penalty
        action_penalty_scale: float = 0.01,     # L2 penalty on actions
        up_bonus_scale: float = 0.5,            # Extra bonus multiplier for upward movement
        stuck_threshold: int = 100,             # Consecutive no-progress steps before penalty
        stuck_penalty: float = 0.1,             # Penalty per step when stuck
        # Ground-stuck (锤子不接地+缸接地) detection
        ground_zone: float = 2.0,               # Y above initial ground = "on the ground"
        ground_stuck_threshold: int = 30,        # Steps stuck on ground before penalty
        ground_stuck_penalty: float = 0.3,       # Penalty per step when stuck on ground
        # Aerial progress bonus (锤子接地+缸离地 -> indirect reward)
        aerial_bonus_scale: float = 0.3,         # Extra bonus when above ground AND rising
    ):
        self.height_scale = height_scale
        self.fall_threshold = fall_threshold
        self.fall_penalty_scale = fall_penalty_scale
        self.action_penalty_scale = action_penalty_scale
        self.up_bonus_scale = up_bonus_scale
        self.stuck_threshold = stuck_threshold
        self.stuck_penalty = stuck_penalty
        self.ground_zone = ground_zone
        self.ground_stuck_threshold = ground_stuck_threshold
        self.ground_stuck_penalty = ground_stuck_penalty
        self.aerial_bonus_scale = aerial_bonus_scale

        # Episode state
        self._prev_y: Optional[float] = None
        self._episode_max_y: float = 0.0
        self._stuck_counter: int = 0
        self._ground_stuck_counter: int = 0
        self._ground_level: float = 0.0

    def reset(self, initial_y: float):
        """Reset reward state for a new episode."""
        self._prev_y = initial_y
        self._episode_max_y = initial_y
        self._ground_level = initial_y  # ground = starting height
        self._stuck_counter = 0
        self._ground_stuck_counter = 0

    def compute(
        self,
        current_y: float,
        action: np.ndarray,
    ) -> Tuple[float, dict, bool]:
        """Compute the reward for a single step.

        Args:
            current_y: Current camera Y coordinate.
            action: The action taken (normalised [dx, dy] in [-1, 1]).

        Returns:
            (reward, reward_components_dict, is_fall)
        """
        prev_y = self._prev_y if self._prev_y is not None else current_y
        height_delta = current_y - prev_y

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ①  Progress reward — ASYMMETRIC (reference style)
        #     Up: full reward.  Down: softly clipped at -0.2.
        #     The reference clips negative progress at -0.2 because
        #     going down is required for the swinging mechanic.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        scaled_delta = height_delta * self.height_scale
        if scaled_delta > 0:
            prog_rew = scaled_delta
        else:
            prog_rew = max(scaled_delta, -0.2)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ②  Upward bonus — reward ANY upward movement
        #     This is our best proxy for "effective hammer use".
        #     The reference has player_vel · hammer_collision for this;
        #     we use upward velocity as a stand-in.
        #     Capped at 1.0 to prevent single-step dominance.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if height_delta > 0:
            up_bonus = min(height_delta * self.height_scale * self.up_bonus_scale, 1.0)
        else:
            up_bonus = 0.0

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ③  Fall penalty — PROPORTIONAL to height lost from peak
        #     The reference penalises downward collision impulse
        #     proportionally.  We penalise height lost from peak.
        #     First fall_threshold units are "free" (normal game swing).
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        drop_from_peak = self._episode_max_y - current_y
        is_fall = drop_from_peak > self.fall_threshold
        if is_fall:
            # Proportional: the further you fall, the worse it is
            fall_pen = -(drop_from_peak - self.fall_threshold) * self.fall_penalty_scale
            self._episode_max_y = current_y  # reset peak after a fall
        else:
            fall_pen = 0.0

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ④  Stuck penalty — nudge agent when it makes zero progress
        #     If |height_delta| is tiny for stuck_threshold consecutive
        #     steps, the agent is probably flailing uselessly.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if abs(height_delta) < 0.005:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        stuck_pen = 0.0
        if self._stuck_counter > self.stuck_threshold:
            stuck_pen = -self.stuck_penalty

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ⑤  Ground-stuck penalty (锤子不接地 + 缸接地)
        #     If the agent is near the ground AND making no progress,
        #     the hammer tip is probably not anchored (just flailing in
        #     the air while sitting on the ground — the most useless state).
        #     Penalise this harder than general stuck to push the agent
        #     toward actually anchoring the hammer and climbing.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        on_ground = current_y < self._ground_level + self.ground_zone
        is_still = abs(height_delta) < 0.005
        if on_ground and is_still:
            self._ground_stuck_counter += 1
        else:
            self._ground_stuck_counter = 0

        ground_stuck_pen = 0.0
        if self._ground_stuck_counter > self.ground_stuck_threshold:
            ground_stuck_pen = -self.ground_stuck_penalty

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ⑥  Aerial progress bonus (锤子接地 + 缸离地 → 间接奖励)
        #     If the agent is ABOVE the ground zone AND height is rising,
        #     that implies: hammer IS anchored AND pot IS lifting.
        #     This is exactly the state we want — give an extra reward.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        off_ground = current_y > self._ground_level + self.ground_zone
        if off_ground and height_delta > 0:
            aerial_bonus = min(height_delta * self.height_scale * self.aerial_bonus_scale, 0.5)
        else:
            aerial_bonus = 0.0

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ⑦  Action penalty — gentle L2 regularisation
        #     Same as reference: -0.01 * ||action||²
        #     Keeps the agent from sending max-delta every step.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        action_l2 = np.sum(action ** 2)
        action_pen = -self.action_penalty_scale * action_l2

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Assemble
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        reward = (
            prog_rew + up_bonus + fall_pen + stuck_pen
            + ground_stuck_pen + aerial_bonus + action_pen
        )

        components = {
            "reward/prog":            prog_rew,
            "reward/up_bonus":         up_bonus,
            "reward/fall":             fall_pen,
            "reward/stuck":            stuck_pen,
            "reward/ground_stuck":     ground_stuck_pen,
            "reward/aerial_bonus":     aerial_bonus,
            "reward/action_l2":        action_pen,
        }

        # Track episode max
        if current_y > self._episode_max_y:
            self._episode_max_y = current_y

        self._prev_y = current_y

        return reward, components, is_fall

    @property
    def episode_max_y(self) -> float:
        return self._episode_max_y
