# labs/ch07-first-gait/04_walk.py
"""7.4 Implementing walk: designing a four-beat phase pattern.

Lateral-sequence walk (LF 0, RH 0.25, RF 0.5, LH 0.75), duty 0.75, 1 Hz. Three feet are always on the ground.
Compare speed and wobble with a trot of the same step length (2 Hz, duty 0.5).
Figures: out/ch07-fig07-walk-frames.png, out/ch07-fig08-walk-vs-trot.png
Run: uv run python ch07-first-gait/04_walk.py   (--view --speed 0.5 recommended)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator
from quadbook.gait import LEG_ORDER
from quadbook.render import play, snapshots, track_camera, want_view
from quadbook.robot import load
from quadbook.sim import WALK_KP, rollout, standard_pd, summary

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CONTROL_EVERY = 5
GAITS = {
    "walk": dict(gait="walk", freq=1.0, duty=0.75, step_length=0.08, step_height=0.04),
    "trot": dict(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04),
}


def run(name, duration=8.0):
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP); gen = GaitGenerator(**GAITS[name])
    return gen, rollout(model, data, gen, pd, duration=duration, control_every=CONTROL_EVERY)


logs = {}
for name in GAITS:
    gen, log = run(name); logs[name] = log; s = summary(log)
    c = log["contact"][log["t"] >= 2.0]
    print(f"{name:4s}: speed {s['speed']:.3f} m/s (ideal {gen.ideal_speed:.3f}) | roll σ {s['roll_std']:.2f}° pitch σ {s['pitch_std']:.2f}° | height {s['height']:.3f} m | mean feet on ground {c.sum(axis=1).mean():.2f} | fell {s['fell']}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for name, log in logs.items():
    axes[0].plot(log["t"], log["x"] - log["x"][0], label=name)
    axes[1].plot(log["t"], np.degrees(log["roll"]), label=name, lw=0.9)
    axes[2].plot(log["t"], np.degrees(log["pitch"]), label=name, lw=0.9)
axes[0].set_title("(a) distance [m]", fontsize=10); axes[1].set_title("(b) torso roll [deg]", fontsize=10); axes[2].set_title("(c) torso pitch [deg]", fontsize=10)
for ax in axes: ax.set_xlabel("time [s]"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch07-fig08-walk-vs-trot.png", dpi=200); plt.close(fig)

# frames: one walk cycle (1 s) in four shots. The swing leg changes from frame to frame
model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP); gen = GaitGenerator(**GAITS["walk"])
cam = track_camera(model, distance=1.3, azimuth=140, elevation=-15)
dt_ctrl = model.opt.timestep * CONTROL_EVERY
state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
def _step():
    if data.time >= 0.5 and state["i"] % CONTROL_EVERY == 0:
        state["q"], state["qd"] = gen.targets_with_velocity(dt_ctrl)
    data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
times = (3.0 + 0.75 + 0.125, 3.0 + 1.0 + 0.125, 3.0 + 1.25 + 0.125, 3.0 + 1.5 + 0.125)   # middle of each leg's swing phase
print("saved", snapshots(model, data, times, OUT / "ch07-fig07-walk-frames.png", camera=cam, step_fn=_step, width=600, height=450))
print("saved", OUT / "ch07-fig08-walk-vs-trot.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP); gen = GaitGenerator(**GAITS["walk"])
    state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    play(model, data, step_fn=_step, duration=10.0, title="walk 1 Hz, duty 0.75 (three feet always on the ground)")
