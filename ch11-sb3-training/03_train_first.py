# labs/ch11-sb3-training/03_train_first.py
"""11.3 Running the first training: keeping the baseline from collapsing.

(A) Short comparison (300k steps each): log_std_init 0 (default) vs -1. Does the baseline gait collapse early in training, and how fast does it recover?
(B) Main training, 3M steps (log_std_init -1, 8 envs, VecNormalize). Every 100k steps, evaluate the deterministic policy for 5 episodes in a separate evaluation env.
    Outputs: out/ppo_first.zip, out/ppo_first_vecnormalize.pkl, out/ppo_first_eval.npz, out/tb_first/ (TensorBoard), out/progress_first.csv
Figure: out/ch11-fig04-logstd-compare.png
Run: uv run python ch11-sb3-training/03_train_first.py   (about 20 to 30 minutes on CPU)
"""

import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure

from common import OUT, PPO_KWARGS, make_envs, normalized


class EpisodeStats(BaseCallback):
    """Collect the episode reward and length recorded by Monitor as (step, value) rows."""
    def __init__(self):
        super().__init__(); self.rows = []
    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.rows.append((self.num_timesteps, info["episode"]["r"], info["episode"]["l"]))
        return True


def train(name, total, log_std_init, eval_every=None):
    venv = normalized(make_envs(n_envs=8, seed=0, subproc=True))
    kw = dict(PPO_KWARGS); kw["policy_kwargs"] = dict(log_std_init=log_std_init)
    model = PPO("MlpPolicy", venv, verbose=0, tensorboard_log=str(OUT / f"tb_{name}"), **kw)
    model.set_logger(configure(str(OUT / f"log_{name}"), ["csv", "tensorboard"]))
    stats = EpisodeStats(); callbacks = [stats]
    if eval_every:
        eval_env = normalized(make_envs(n_envs=1, seed=999, subproc=False), training=False)
        callbacks.append(EvalCallback(eval_env, eval_freq=eval_every // 8, n_eval_episodes=5, deterministic=True,
                                      log_path=str(OUT / f"eval_{name}"), best_model_save_path=str(OUT / f"best_{name}"), verbose=0))
    t0 = time.perf_counter()
    model.learn(total_timesteps=total, callback=callbacks)
    el = time.perf_counter() - t0
    model.save(str(OUT / f"{name}.zip")); venv.save(str(OUT / f"{name}_vecnormalize.pkl"))
    shutil.copy(OUT / f"log_{name}" / "progress.csv", OUT / f"progress_{name}.csv")
    venv.close()
    return np.array(stats.rows), el


if __name__ == "__main__":
    print("(A) log_std_init comparison, 300k steps each")
    curves = {}
    for ls in (0.0, -1.0):
        rows, el = train(f"cmp_logstd{int(ls)}", 300_000, ls)
        curves[ls] = rows
        first = rows[rows[:, 0] <= 40_000]; last = rows[rows[:, 0] >= 260_000]
        print(f"    log_std_init {ls:+.0f}: {el/60:.1f} min | first 40k steps: mean episode reward {first[:,1].mean():.1f}, length {first[:,2].mean():.0f} | last 40k: reward {last[:,1].mean():.1f}, length {last[:,2].mean():.0f} (baseline a=0: reward 249, length 1000)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ls, rows in curves.items():
        order = np.argsort(rows[:, 0]); r = rows[order]
        k = 20; sm = lambda v: np.convolve(v, np.ones(k) / k, mode="valid")
        axes[0].plot(r[k - 1:, 0], sm(r[:, 1]), label=f"log_std_init {ls:+.0f}"); axes[1].plot(r[k - 1:, 0], sm(r[:, 2]), label=f"log_std_init {ls:+.0f}")
    axes[0].axhline(249, color="#999999", ls=":", label="baseline (a = 0)"); axes[0].set_xlabel("timesteps"); axes[0].set_ylabel("episode return (raw)"); axes[0].set_title("(a) episode return during early training", fontsize=10); axes[0].legend(fontsize=8)
    axes[1].axhline(1000, color="#999999", ls=":"); axes[1].set_xlabel("timesteps"); axes[1].set_ylabel("episode length [steps]"); axes[1].set_title("(b) episode length (1000 = never fell)", fontsize=10); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "ch11-fig04-logstd-compare.png", dpi=200); plt.close(fig)
    print("saved", OUT / "ch11-fig04-logstd-compare.png")

    print("(B) Main training, 3M steps (log_std_init -1)")
    rows, el = train("ppo_first", 3_000_000, -1.0, eval_every=100_000)
    ev = np.load(OUT / "eval_ppo_first" / "evaluations.npz")
    print(f"    {el/60:.1f} min | evaluation return (deterministic, 5 episodes): first {ev['results'][0].mean():.1f} -> best {ev['results'].mean(axis=1).max():.1f} (at {ev['timesteps'][ev['results'].mean(axis=1).argmax()]:,} steps) -> last {ev['results'][-1].mean():.1f} | last evaluation length {ev['ep_lengths'][-1].mean():.0f}")
    np.savez(OUT / "ppo_first_eval.npz", timesteps=ev["timesteps"], results=ev["results"], ep_lengths=ev["ep_lengths"], train_rows=rows)
    print("saved", OUT / "ppo_first.zip", OUT / "ppo_first_vecnormalize.pkl", OUT / "ppo_first_eval.npz")
