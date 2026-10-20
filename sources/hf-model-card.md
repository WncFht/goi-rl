---
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
tags:
- reinforcement-learning
- deep-reinforcement-learning
- stable-baselines3
- ppo
- getting-over-it
---

# PPO agents for Getting Over It with Bennett Foddy

**Goal: be the first to finish the real *Getting Over It* with reinforcement learning.** That means neural networks
playing the actual Steam game (not a clone) in one continuous run from the bottom to the top, with no save/load and no
search while playing.

**Status (October 2026):** playing alone from the bottom, the networks reach the anvil (about 530 of the route's 864
units, best 542). Started from the mast (745), they finish 8 times out of 8. A separate search agent with save/load
has finished the game, but that doesn't count as the RL result.

**Video (`replay.mp4`):** 30 s in real time of the networks climbing alone through the boxes and the furniture to the
orange stretch (route progress 308 to 405), with their mouse input in the corner. It starts from a state saved in one
of their continuous games from the bottom; 2 of 11 tries from that state climbed this fast.

**Files:** 27 PPO policies (Stable-Baselines3). Most were trained on one stretch of the route, starting episodes from
states along climbs that had already succeeded. `chain.json` lists which policy plays from which route progress, with
its exploration-noise scale (`@0.5`, `@1`). They play inside the real game through our BepInEx plugin, which exposes the
game to Python over a local socket.

**Code:** https://github.com/miwate/getting-over-it-rl
