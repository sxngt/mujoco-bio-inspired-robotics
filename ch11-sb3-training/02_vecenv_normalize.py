# labs/ch11-sb3-training/02_vecenv_normalize.py
"""11.2 Parallel environments (VecEnv) and normalization (VecNormalize).

(a) Throughput of 1 env, 8 envs in DummyVecEnv (one process), and 8 envs in SubprocVecEnv (8 processes).
(b) How VecNormalize equalizes the scale of each observation group: std before and after normalization. The scale of reward normalization.
Figures: out/ch11-fig02-throughput.png, out/ch11-fig03-normalization.png
Run: uv run python ch11-sb3-training/02_vecenv_normalize.py
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT, make_envs, normalized

GROUPS = [("joint offset", 0, 12), ("joint vel", 12, 24), ("gravity", 24, 27), ("gyro", 27, 30), ("lin vel", 30, 33), ("phase", 33, 41), ("prev action", 41, 53)]


def throughput(venv, steps=400):
    venv.reset(); n = venv.num_envs; t0 = time.perf_counter()
    for _ in range(steps):
        venv.step(np.random.uniform(-1, 1, (n, 12)))
    return steps * n / (time.perf_counter() - t0)


if __name__ == "__main__":
    rates = {}
    rates["1 env"] = throughput(make_envs(n_envs=1, subproc=False))
    rates["8 envs, one process"] = throughput(make_envs(n_envs=8, subproc=False))
    sub = make_envs(n_envs=8, subproc=True); rates["8 envs, 8 processes"] = throughput(sub); sub.close()
    print("(a) Throughput [control steps/s]")
    for k, v in rates.items(): print(f"    {k:22s} {v:8,.0f}  ({v*0.01:.0f}x real time)")
    fig, ax = plt.subplots(figsize=(6, 3.2)); ax.bar(list(rates), list(rates.values()), color="#4a90d9"); ax.set_ylabel("control steps / s"); ax.set_title("environment throughput", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "ch11-fig02-throughput.png", dpi=200); plt.close(fig)

    # (b) Normalization
    raw = make_envs(n_envs=8, subproc=False); venv = normalized(raw)
    obs = venv.reset(); raw_obs, norm_obs, raw_r, norm_r = [], [], [], []
    for _ in range(1500):
        a = np.random.uniform(-0.3, 0.3, (8, 12))
        obs, r, done, info = venv.step(a)
        norm_obs.append(obs.copy()); raw_obs.append(venv.get_original_obs().copy()); norm_r.append(r.copy()); raw_r.append(venv.get_original_reward().copy())
    raw_obs, norm_obs = np.concatenate(raw_obs), np.concatenate(norm_obs); raw_r, norm_r = np.concatenate(raw_r), np.concatenate(norm_r)
    print("(b) VecNormalize statistics after 12,000 steps: std per group (raw observation -> normalized observation)")
    for name, a, b in GROUPS:
        print(f"    {name:12s} {raw_obs[:, a:b].std():.3f} -> {norm_obs[:, a:b].std():.3f}")
    print(f"    reward: raw reward sigma {raw_r.std():.3f}, normalized reward sigma {norm_r.std():.3f} (divided by the running std of the return)")
    fig, ax = plt.subplots(figsize=(8, 3.4)); x = np.arange(len(GROUPS))
    ax.bar(x - 0.2, [raw_obs[:, a:b].std() for _, a, b in GROUPS], 0.4, label="raw observation"); ax.bar(x + 0.2, [norm_obs[:, a:b].std() for _, a, b in GROUPS], 0.4, label="after VecNormalize")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in GROUPS], fontsize=8); ax.set_yscale("log"); ax.set_ylabel("std (log)"); ax.set_title("VecNormalize equalizes observation groups", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "ch11-fig03-normalization.png", dpi=200); plt.close(fig)
    print("saved", OUT / "ch11-fig02-throughput.png", OUT / "ch11-fig03-normalization.png")
