# Chapter 12. Sculpting the Reward

`train_lib.py` holds the reward configs (`CONFIGS`: R1 to R4 controlled experiments, H1 to H4 hacking museum, C_scratch/C_curriculum curriculum), `train()`, and `evaluate()` (Chapter 8 metrics + push limit → `out/metrics_<name>.json`).
`00_train_all.py` runs every training in order (14 configs, about 2 hours, CPU) and evaluates each right away. Existing results are skipped. Scripts 01 to 05 read those results.

| Script | Section | Contents |
|---|---|---|
| `00_train_all.py` | all | Train + evaluate 10 configs (`uv run python 00_train_all.py [name ...]`) |
| `01_controlled.py` | 12.1 | R1 with a single heading term added vs the Chapter 11 policy: top-view paths, metric bars |
| `02_terms.py` | 12.2 | R1 → R2 (speed tracking) → R3 (contact match): speed over time, gait diagrams |
| `03_hacking.py` | 12.3 | Scenes and metrics of H1 (forward only), H2 (alive + energy), H3 (impact heavy), H4 (contact heavy) (replay with `--view H1` etc.) |
| `04_curriculum.py` | 12.4 | Six disturbance-training conditions (scratch 60 N, constant 20 N, 3M ramp, 6M ramp + hold, 80 N ramp, 0.15 correction range): learning curves, push survival rates (`out/push_survival.json`) |
| `05_final.py` | 12.5 | Metrics for the whole chain, and for the final policy (the most push-resistant curriculum among those that kept the gait = C_curriculum_6M): Froude coordinates, gait diagram against a dog trot, frames (replay with `--view`) |

Reads the Chapter 11 outputs (`../ch11-sb3-training/out/ppo_first*`), so Chapter 11's script 03 must be run first. The new reward terms and random pushes live in `quadbook/env.py` (`DEFAULT_WEIGHTS`, `push_max`).

Note: change environment attributes inside a VecEnv through a method, as in `env_method("set_push_max", v)`. `set_attr`/`__setattr__` only reach the Monitor wrapper and do not propagate to the environment.
