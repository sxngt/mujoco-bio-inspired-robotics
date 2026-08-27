# Chapter 13. Into the Wider World

`train_lib.py` holds the three Chapter 13 training configs (observation without linear velocity, residual only at a 0.8 m/s target, generator opened) and the evaluation helpers (a copy of the Chapter 12 train_lib; no imports across chapter folders).
It reads the Chapter 12 outputs (`../ch12-reward-shaping/out/C_curriculum_6M*`, `R4_full*`). The MJX benchmark runs after `uv sync --extra mjx`.

| Script | Section | Contents |
|---|---|---|
| `00_train_all.py` | 13.1, 13.3 | Train and evaluate A_no_linvel, B_residual_08, B_modulate_08 (about 20 min) |
| `01_robustness.py` | 13.1 | Final policy vs open-loop: rough terrain (`models/quadruped_rough.xml`), slope, friction, payload |
| `02_mjx_benchmark.py` | 13.2 | The same XML on MJX, throughput per batch size (CPU JAX) |
| `03_sim2real_gaps.py` | 13.3 | Observation noise, action latency, and a policy trained without body linear velocity |
| `04_open_generator.py` | 13.1 | Generator frequency and step length opened to the policy: speed trace, parameters chosen by the policy, dynamic similarity coordinates |
| `05_slope_anatomy.py` | 13.3 | The policy on a gravity-tilted slope vs a floor-tilted slope (`models/quadruped_slope*.xml`) |

Chapter 13 environment options (`quadbook/env.py`): `use_lin_vel`, `modulate_generator` (14-dim action), `model_file`, `obs_noise`, `latency_steps`.
Run from labs/ with `cd ch13-wider-world && uv run python 00_train_all.py`, then 01~05. `01 --view --speed 0.5` shows the rough terrain (use `mjpython` on macOS).
