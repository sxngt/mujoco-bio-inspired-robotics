# Chapter 10. Building the Environment: Gymnasium

This chapter produces `QuadrupedEnv` in `quadbook/env.py` (a Gymnasium environment, id `QuadBook/Quadruped-v0`), which Chapters 11 and 12 use for training as is.
Action = Chapter 7 generator target + 0.1 rad offset, observation of 53 dimensions, reward = weighted sum (terms in `info["terms"]`), terminated on a fall, truncated at 10 s.

| Script | Section | Contents |
|---|---|---|
| `01_env_interface.py` | 10.1 | Spaces, env_checker, action 0 vs random action, throughput (steps per second) |
| `02_observation.py` | 10.2 | Traces and scales of the 53-dimensional observation by group |
| `03_action.py` | 10.3 | Sweep of the residual scale 0.02 to 0.4 rad, comparison with absolute mode (action = joint target) |
| `04_reward.py` | 10.4 | Raw magnitude of the reward terms under four behaviours and their contribution after weighting |
| `05_termination.py` | 10.5 | Episode length distribution (terminated/truncated), the scene at termination, reset randomization |

Run from labs/ with `uv run python ch10-gym-environment/<script>.py`; add `--view --speed 0.5` to watch the scene (use `mjpython` on macOS).
