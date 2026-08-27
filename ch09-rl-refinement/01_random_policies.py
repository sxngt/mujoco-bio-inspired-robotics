# labs/ch09-rl-refinement/01_random_policies.py
"""9.3 Side experiment: how far does a 'random policy' walk before any training?

Reinforcement learning starts from random exploration. Three starting points are compared over 20 episodes each (4 s per episode).
(a) Blank-slate torque: at every control step, 12 torques drawn uniformly in ±12 N·m.
(b) Blank-slate joint targets: standing pose ± 0.3 rad uniform noise as the PD target.
(c) CPG + residual: the Chapter 7 trot generator target plus ± 0.1 rad uniform noise as the PD target (the action design of Section 10.3).
Metrics: forward distance in 4 s, fraction of episodes that fell, mean forward speed (a proxy for the reward signal).
Figure: out/ch09-fig04-random-policies.png
Run: uv run python ch09-rl-refinement/01_random_policies.py   (--view --speed 0.5: watch one episode of (c))
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator
from quadbook.render import play, want_view
from quadbook.robot import STAND_POSE_BALANCED, load
from quadbook.sim import WALK_KP, fallen, standard_pd

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CE, T, N_EP = 5, 4.0, 20
rng = np.random.default_rng(0)


def episode(kind, seed):
    r = np.random.default_rng(seed)
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04)
    dt_ctrl = model.opt.timestep * CE
    q, qd = STAND_POSE_BALANCED.copy(), np.zeros(12); tau = np.zeros(12); i = 0
    x0 = data.body("torso").xpos[0]; fell = False; t_fall = T
    while data.time < T:
        if i % CE == 0:
            if kind == "random torque":
                tau = r.uniform(-12, 12, 12)
            elif kind == "random targets":
                q = STAND_POSE_BALANCED + r.uniform(-0.3, 0.3, 12); qd = np.zeros(12)
            else:
                q, qd = gen.targets_with_velocity(dt_ctrl); q = q + r.uniform(-0.1, 0.1, 12)
        data.ctrl[:] = tau if kind == "random torque" else pd.torque(data, q, qd)
        mujoco.mj_step(model, data); i += 1
        if not fell and fallen(data):
            fell, t_fall = True, data.time
    return data.body("torso").xpos[0] - x0, fell, t_fall


kinds = ("random torque", "random targets", "CPG + random residual")
res = {}
print(f"Random policies, {N_EP} episodes × {T:.0f} s")
for kind in kinds:
    d, f, tf = zip(*[episode(kind, s) for s in range(N_EP)])
    res[kind] = (np.array(d), np.array(f), np.array(tf))
    print(f"  {kind:22s}: mean forward distance {np.mean(d):+.2f} m (min {np.min(d):+.2f}, max {np.max(d):+.2f}) | fell {100*np.mean(f):.0f}% | mean time to fall {np.mean(tf):.1f} s | mean speed {np.mean(d)/T:+.3f} m/s")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
colors = ("#d62728", "#ff7f0e", "#2ca02c")
for k, (kind, c) in enumerate(zip(kinds, colors)):
    d = res[kind][0]
    axes[0].scatter(np.full(N_EP, k) + rng.uniform(-0.15, 0.15, N_EP), d, color=c, s=18, alpha=0.8)
    axes[0].plot([k - 0.25, k + 0.25], [d.mean()] * 2, color="#333333", lw=2)
axes[0].axhline(0, color="#999999", ls=":"); axes[0].set_xticks(range(3)); axes[0].set_xticklabels(kinds, fontsize=8)
axes[0].set_ylabel("forward distance in 4 s [m]"); axes[0].set_title("(a) distance per episode (bar = mean)", fontsize=10)
axes[1].bar(kinds, [100 * res[k][1].mean() for k in kinds], color=colors)
axes[1].set_ylabel("episodes that fell [%]"); axes[1].set_ylim(0, 105); axes[1].set_title("(b) fall rate", fontsize=10); axes[1].tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch09-fig04-random-policies.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch09-fig04-random-policies.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04)
    r = np.random.default_rng(1); state = {"i": 0, "q": STAND_POSE_BALANCED.copy(), "qd": np.zeros(12)}
    def _step():
        if state["i"] % CE == 0:
            q, qd = gen.targets_with_velocity(model.opt.timestep * CE); state["q"], state["qd"] = q + r.uniform(-0.1, 0.1, 12), qd
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
    play(model, data, step_fn=_step, duration=8.0, title="CPG + random residual (±0.1 rad): it walks even before training")
