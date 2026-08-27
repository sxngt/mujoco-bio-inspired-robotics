# labs/ch07-first-gait/05_param_sweep.py
"""7.5 Parameter experiments: frequency, amplitude (step length·step height), phase differences.

(a) Trot with frequency {1, 2, 3} Hz × step length {0.04, 0.08, 0.12} m: measured speed vs ideal speed L·f/β.
(b) Step height {0.01, 0.04, 0.08} m: foot dragging and bouncing.
(c) Changing the phase table: trot, pace, bound, walk (phases only, duty 0.5), pronk (all 0). Which ones survive at the same f·L·duty?
Figures: out/ch07-fig09-speed-sweep.png, out/ch07-fig10-phase-sweep.png
Run: uv run python ch07-first-gait/05_param_sweep.py   (--view: watch the bound from (c))
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator
from quadbook.gait import GAIT_PHASES
from quadbook.render import play, want_view
from quadbook.robot import load
from quadbook.sim import WALK_KP, rollout, standard_pd, summary

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def trial(duration=6.0, **kw):
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP); gen = GaitGenerator(**kw)
    log = rollout(model, data, gen, pd, duration=duration, control_every=5)
    return gen, summary(log)


# ---- (a) frequency × step length ----
freqs, lengths = (1.0, 2.0, 3.0), (0.04, 0.08, 0.12)
print("(a) trot, duty 0.5, step height 0.04: measured speed / ideal speed [m/s]")
meas = np.zeros((len(freqs), len(lengths))); ideal = np.zeros_like(meas); fell = np.zeros_like(meas, dtype=bool)
for i, f in enumerate(freqs):
    row = []
    for j, L in enumerate(lengths):
        gen, s = trial(gait="trot", freq=f, duty=0.5, step_length=L, step_height=0.04)
        meas[i, j], ideal[i, j], fell[i, j] = s["speed"], gen.ideal_speed, s["fell"]
        row.append(f"{s['speed']:.2f}/{gen.ideal_speed:.2f}" + ("!" if s["fell"] else " "))
    print(f"    f = {f:.0f} Hz: " + "   ".join(f"L {L:.2f}: {r}" for L, r in zip(lengths, row)))
print("    (! = fell)")

# ---- (b) step height ----
print("(b) trot 2 Hz, step length 0.08: effect of step height")
for h in (0.01, 0.04, 0.08):
    gen, s = trial(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=h)
    print(f"    h = {h:.2f} m: speed {s['speed']:.3f} m/s ({100*s['speed']/gen.ideal_speed:.0f}%), pitch σ {s['pitch_std']:.2f}°, fell {s['fell']}")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
for j, L in enumerate(lengths):
    ax.plot(ideal[:, j], meas[:, j], "o-", label=f"step length {L:.2f} m")
    for i in range(len(freqs)):
        if fell[i, j]: ax.plot(ideal[i, j], meas[i, j], "x", color="#d62728", ms=10)
lim = ideal.max() * 1.05
ax.plot([0, lim], [0, lim], ls=":", color="#999999", label="measured = ideal")
ax.set_xlabel("ideal speed L f / beta [m/s]"); ax.set_ylabel("measured speed [m/s]"); ax.set_title("trot: frequency x step length (x = fell)", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch07-fig09-speed-sweep.png", dpi=200); plt.close(fig)

# ---- (c) phase table ----
patterns = {"trot": GAIT_PHASES["trot"], "pace": GAIT_PHASES["pace"], "bound": GAIT_PHASES["bound"],
            "walk phases": GAIT_PHASES["walk"], "pronk": {"LF": 0, "RF": 0, "LH": 0, "RH": 0}}
print("(c) same f 2 Hz, duty 0.5, step length 0.08, changing only the phase table")
res = {}
for name, ph in patterns.items():
    gen, s = trial(offsets=ph, freq=2.0, duty=0.5, step_length=0.08, step_height=0.04)
    res[name] = s
    print(f"    {name:12s}: speed {s['speed']:+.3f} m/s, roll σ {s['roll_std']:5.2f}°, pitch σ {s['pitch_std']:5.2f}°, height {s['height']:.3f}, fell {s['fell']}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
names = list(res)
axes[0].bar(names, [res[n]["speed"] for n in names], color=["#d62728" if res[n]["fell"] else "#4a90d9" for n in names])
axes[0].set_ylabel("speed [m/s]"); axes[0].set_title("(a) speed (red = fell)", fontsize=10)
axes[1].bar(names, [res[n]["roll_std"] for n in names], label="roll", alpha=0.8); axes[1].bar(names, [res[n]["pitch_std"] for n in names], label="pitch", alpha=0.6)
axes[1].set_ylabel("std [deg]"); axes[1].set_title("(b) torso attitude wobble", fontsize=10); axes[1].legend(fontsize=8)
for ax in axes: ax.tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch07-fig10-phase-sweep.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch07-fig09-speed-sweep.png", OUT / "ch07-fig10-phase-sweep.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(offsets=GAIT_PHASES["bound"], freq=2.0, duty=0.5, step_length=0.08, step_height=0.04)
    state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    def _step():
        if data.time >= 0.5 and state["i"] % 5 == 0: state["q"], state["qd"] = gen.targets_with_velocity(model.opt.timestep * 5)
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
    play(model, data, step_fn=_step, duration=8.0, title="bound phase table (front pair, hind pair)")
