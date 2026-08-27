# labs/ch10-gym-environment/02_observation.py
"""10.2 Observation design: what does the robot need to know?

Run one episode with action 0, record the 53-dimensional observation, and look at what each group shows and how large its values are (scale).
Observations with such uneven scales are why Section 11.2 needs normalization (VecNormalize).
Figures: out/ch10-fig03-observation-traces.png, out/ch10-fig04-observation-scale.png
Run: uv run python ch10-gym-environment/02_observation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadbook.env import QuadrupedEnv

OUT = Path(__file__).resolve().parent / "out"
GROUPS = [("joint offset", 0, 12), ("joint vel x0.05", 12, 24), ("gravity (body)", 24, 27), ("gyro x0.25", 27, 30), ("lin vel (body)", 30, 33), ("CPG phase sin/cos", 33, 41), ("prev action", 41, 53)]

env = QuadrupedEnv(random_init_phase=False)
obs, _ = env.reset(seed=0)
O, T = [obs], [0.0]
while True:
    obs, r, term, trunc, info = env.step(np.zeros(12)); O.append(obs); T.append(info["t"])
    if term or trunc: break
O, T = np.array(O), np.array(T)
print(f"observation of {O.shape[1]} dimensions, {len(T)} steps")
np.set_printoptions(precision=2, suppress=True, linewidth=150)
print("observation vector at t = 3 s:"); k = np.argmin(np.abs(T - 3.0))
for name, a, b in GROUPS:
    print(f"  {name:18s} {O[k, a:b]}")
print("value magnitude by group (std over the episode / max absolute value)")
for name, a, b in GROUPS:
    print(f"  {name:18s} σ {O[:, a:b].std():.3f}  max|·| {np.abs(O[:, a:b]).max():.3f}")

m = (T >= 2.0) & (T <= 3.0)
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
axes[0, 0].plot(T[m], O[m, 24:27]); axes[0, 0].set_title("(a) gravity vector in torso frame (x, y, z)", fontsize=10); axes[0, 0].legend(["x", "y", "z"], fontsize=8)
axes[0, 1].plot(T[m], O[m, 33:37]); axes[0, 1].set_title("(b) sin of CPG phase, four legs", fontsize=10); axes[0, 1].legend(["LF", "RF", "LH", "RH"], fontsize=8)
axes[1, 0].plot(T[m], O[m, 1:3]); axes[1, 0].plot(T[m], O[m, 4:6], ls="--"); axes[1, 0].set_title("(c) joint offsets: LF hip, LF knee (solid), RF hip, RF knee (dashed)", fontsize=10)
axes[1, 1].plot(T[m], O[m, 30:33]); axes[1, 1].set_title("(d) torso linear velocity in body frame (x, y, z)", fontsize=10); axes[1, 1].legend(["vx", "vy", "vz"], fontsize=8)
for ax in axes.flat: ax.set_xlabel("time [s]")
fig.tight_layout(); fig.savefig(OUT / "ch10-fig03-observation-traces.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.4))
names = [g[0] for g in GROUPS]; stds = [O[:, a:b].std() for _, a, b in GROUPS]; mx = [np.abs(O[:, a:b]).max() for _, a, b in GROUPS]
x = np.arange(len(names)); ax.bar(x - 0.2, stds, 0.4, label="std over episode"); ax.bar(x + 0.2, mx, 0.4, label="max |value|")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=15); ax.set_yscale("log"); ax.set_ylabel("value (log)"); ax.set_title("observation groups have different scales (after the fixed scaling in env.py)", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch10-fig04-observation-scale.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch10-fig03-observation-traces.png", OUT / "ch10-fig04-observation-scale.png")
