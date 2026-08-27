# labs/ch10-gym-environment/04_reward.py
"""10.4 First steps in reward design: forward progress, attitude, energy.

Measure the 'raw' magnitude of each reward term under four behaviours: action 0 (trot), random 0.1, random 0.4, standing still (freq 0).
You need the magnitudes before you can choose weights. After weighting, the per-term contributions are placed side by side.
Figure: out/ch10-fig06-reward-terms.png
Run: uv run python ch10-gym-environment/04_reward.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadbook.env import DEFAULT_WEIGHTS, QuadrupedEnv

OUT = Path(__file__).resolve().parent / "out"
TERMS = ("forward", "attitude", "energy", "alive")


def measure(env, policy, seed=0):
    rng = np.random.default_rng(seed); obs, _ = env.reset(seed=seed)
    acc = {k: [] for k in TERMS}; ret = 0.0; n = 0
    while True:
        obs, r, term, trunc, info = env.step(policy(rng)); ret += r; n += 1
        for k in TERMS: acc[k].append(info["terms"][k])
        if term or trunc: break
    return {k: float(np.mean(v)) for k, v in acc.items()}, ret, n


behaviours = {
    "CPG trot (a = 0)": (QuadrupedEnv(), lambda rng: np.zeros(12)),
    "random 0.1 rad": (QuadrupedEnv(action_scale=0.1), lambda rng: rng.uniform(-1, 1, 12)),
    "random 0.4 rad": (QuadrupedEnv(action_scale=0.4), lambda rng: rng.uniform(-1, 1, 12)),
    "stand still (freq 0)": (QuadrupedEnv(freq=0.0, random_init_phase=False), lambda rng: np.zeros(12)),
}
print("per-step mean of reward terms (raw)     forward[m/s]  attitude[rad²]  energy[W]   | weighted reward per step / episode return / steps")
print("weights:", DEFAULT_WEIGHTS)
res = {}
for name, (env, pol) in behaviours.items():
    terms, ret, n = measure(env, pol); res[name] = (terms, ret, n)
    print(f"  {name:22s} {terms['forward']:+8.3f}     {terms['attitude']:+9.4f}     {terms['energy']:+8.2f}   | {ret/n:+.3f} / {ret:+.1f} / {n}")

fig, ax = plt.subplots(figsize=(8.5, 3.6))
names = list(res); x = np.arange(len(names)); w = 0.2
for i, k in enumerate(("forward", "attitude", "energy")):
    ax.bar(x + (i - 1) * w, [DEFAULT_WEIGHTS[k] * res[n][0][k] for n in names], w, label=f"{k} x {DEFAULT_WEIGHTS[k]}")
ax.plot(x, [res[n][1] / res[n][2] for n in names], "k_", ms=25, mew=2, label="total per step")
ax.axhline(0, color="#999999", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel("weighted reward per step"); ax.set_title("reward terms after weighting, four behaviours", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch10-fig06-reward-terms.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch10-fig06-reward-terms.png")
