# labs/ch11-sb3-training/04_curves.py
"""11.4 How to read training curves: together with TensorBoard.

Read progress_ppo_first.csv (SB3 logger) and ppo_first_eval.npz (evaluation callback) left by 03, plot six curves, and summarize how to read each one.
To see the same thing in TensorBoard: uv run tensorboard --logdir ch11-sb3-training/out/tb_ppo_first
Figures: out/ch11-fig05-training-curves.png, out/ch11-fig06-eval-curve.png
Run: uv run python ch11-sb3-training/04_curves.py
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT

rows = list(csv.DictReader(open(OUT / "progress_ppo_first.csv")))
def col(name):
    xs, ys = [], []
    for r in rows:
        if r.get(name, "") not in ("", None):
            xs.append(float(r["time/total_timesteps"])); ys.append(float(r[name]))
    return np.array(xs), np.array(ys)

panels = [
    ("rollout/ep_rew_mean", "(a) episode return (raw reward)", 249.0),
    ("rollout/ep_len_mean", "(b) episode length (1000 = never fell)", 1000.0),
    ("train/approx_kl", "(c) approx KL between old and new policy", None),
    ("train/clip_fraction", "(d) fraction of samples clipped (ratio outside 1 +- 0.2)", None),
    ("train/explained_variance", "(e) explained variance of the value function", None),
    ("train/std", "(f) action std (exploration noise)", None),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, (name, title, ref) in zip(axes.flat, panels):
    x, y = col(name); ax.plot(x / 1e6, y, lw=1)
    if ref is not None: ax.axhline(ref, color="#999999", ls=":", label="baseline (a = 0)"); ax.legend(fontsize=8)
    ax.set_title(title, fontsize=9); ax.set_xlabel("timesteps [M]")
fig.tight_layout(); fig.savefig(OUT / "ch11-fig05-training-curves.png", dpi=200); plt.close(fig)

for name, title, _ in panels:
    x, y = col(name)
    print(f"{name:26s} first {y[0]:8.3f}  1M {y[np.argmin(np.abs(x-1e6))]:8.3f}  2M {y[np.argmin(np.abs(x-2e6))]:8.3f}  last {y[-1]:8.3f}")

ev = np.load(OUT / "ppo_first_eval.npz")
m, s = ev["results"].mean(axis=1), ev["results"].std(axis=1)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(ev["timesteps"] / 1e6, m, "o-", ms=3, label="deterministic policy, 5 episodes"); ax.fill_between(ev["timesteps"] / 1e6, m - s, m + s, alpha=0.2)
ax.axhline(249, color="#999999", ls=":", label="baseline (a = 0): 249"); ax.set_xlabel("timesteps [M]"); ax.set_ylabel("episode return"); ax.set_title("evaluation curve (separate env, no exploration noise)", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch11-fig06-eval-curve.png", dpi=200); plt.close(fig)
print(f"Evaluation curve: first {m[0]:.1f} -> best {m.max():.1f} (at {ev['timesteps'][m.argmax()]/1e6:.1f}M) -> last {m[-1]:.1f} +- {s[-1]:.1f}")
print("saved", OUT / "ch11-fig05-training-curves.png", OUT / "ch11-fig06-eval-curve.png")
