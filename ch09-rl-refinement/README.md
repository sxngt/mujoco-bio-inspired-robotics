# Chapter 9. Reinforcement Learning as a Refinement Tool

Chapter 9 is a conceptual chapter with no code of its own. This folder holds a single side experiment (how far a random policy walks before training). The diagram scripts for the chapter text live in labs_dev/figures. The training code is in Chapters 10 to 12.

| Script | Section | Contents |
|---|---|---|
| `01_random_policies.py` | 9.3 | Random torque / random joint targets / CPG + random residual, forward distance and fall rate over 20 episodes each |

Run from labs/ with `uv run python ch09-rl-refinement/<script>.py`. `01 --view --speed 0.5` plays a CPG + random residual episode (use `mjpython` on macOS).
