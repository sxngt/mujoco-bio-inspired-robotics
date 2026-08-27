# labs/ch12-reward-shaping/01_controlled.py
"""12.1 Adding reward terms one at a time: the craft of the controlled experiment.

Same environment, same seed, same steps (3M): R1 adds a single reward term (heading 1.0, lateral 0.5) and is compared with the Chapter 11 policy.
Figures: out/ch12-fig01-heading-paths.png, out/ch12-fig02-controlled-bars.png
Run: uv run python 01_controlled.py   (00_train_all.py must have been run first)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_lib import OUT, load_policy, run_episode

M = {n: json.load(open(OUT / f"metrics_{n}.json")) for n in ("baseline", "ppo_first", "R1_heading")}
print("5-seed mean (10 s episodes)          speed[m/s]  yaw[°]   y[m]   fragments/cycle(LF RF LH RH)   impact[N]  CoT   margin[cm]  push limit[N]")
for n, m in M.items():
    print(f"  {n:16s} {m['speed']:8.3f} {m['yaw_deg']:+7.0f} {m['y']:+6.2f}   {[round(f, 1) for f in m['fragments']]}   {m['impact']:5.0f}  {m['cot']:.2f}  {100*m['margin']:+6.1f}   {m['push_limit']}")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
for n, c in (("ppo_first", "#ff7f0e"), ("R1_heading", "#2ca02c")):
    model, venv, env = load_policy(n)
    for s in (7, 8, 9):
        lg = run_episode(venv, env, lambda o: model.predict(o, deterministic=True)[0], seed=s)
        ax.plot(lg["x"], lg["y"], color=c, lw=1.2, alpha=0.8, label=(n + " (ch11 reward)" if n == "ppo_first" else n + " (+ heading, lateral)") if s == 7 else None)
    venv.close()
ax.axhline(0, color="#999999", ls=":"); ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("top view, three seeds each: one reward term fixes the veering", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "ch12-fig01-heading-paths.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
names = list(M); cols = ("#999999", "#ff7f0e", "#2ca02c")
axes[0].bar(names, [M[n]["speed"] for n in names], color=cols); axes[0].set_title("speed [m/s]", fontsize=10)
axes[1].bar(names, [abs(M[n]["yaw_deg"]) for n in names], color=cols); axes[1].set_title("|yaw| after 10 s [deg]", fontsize=10)
axes[2].bar(names, [np.mean(M[n]["fragments"][2:]) for n in names], color=cols); axes[2].set_title("hind-leg contact fragments per cycle", fontsize=10)
for ax in axes: ax.tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch12-fig02-controlled-bars.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch12-fig01-heading-paths.png", OUT / "ch12-fig02-controlled-bars.png")
