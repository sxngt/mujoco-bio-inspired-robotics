# Chapter 11. Training with SB3

`common.py` holds the PPO settings of Table 9-2 (`PPO_KWARGS`) and the VecEnv helpers (`make_envs`, `normalized`, `load_trained`). Chapter 12 uses it as is.
03 trains and leaves `out/ppo_first.zip`, `out/ppo_first_vecnormalize.pkl`, `out/progress_ppo_first.csv`, `out/ppo_first_eval.npz`, and `out/tb_ppo_first/`; 04 and 05 read them.

| Script | Section | Contents |
|---|---|---|
| `01_ppo_setup.py` | 11.1 | Policy network structure, parameter count, data per update; untrained action distribution for log_std_init 0 vs -1 |
| `02_vecenv_normalize.py` | 11.2 | Throughput of 1 env / 8 envs in one process / 8 envs in 8 processes; observation and reward scale before and after VecNormalize |
| `03_train_first.py` | 11.3 | log_std_init comparison (300k each) + main training for 3M steps (about 8 min on CPU), deterministic evaluation every 100k |
| `04_curves.py` | 11.4 | Six curves from progress.csv (return, length, KL, clip fraction, explained variance, std) and the evaluation curve |
| `05_compare.py` | 11.5 | Baseline vs learned policy: speed, CoT, Chapter 8 gait_metrics, stability margin, touchdown impact, shape of delta, top-view path, push |

Run from labs/ with `cd ch11-sb3-training && uv run python 03_train_first.py`, in order (03 -> 04 -> 05). TensorBoard: `uv run tensorboard --logdir out/tb_ppo_first`. Use `05 --view --speed 0.5` to watch the learned policy (`mjpython` on macOS).
