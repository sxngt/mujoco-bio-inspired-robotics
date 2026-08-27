# labs/ch07-first-gait/03_trot.py
"""7.3 Implementing trot: synchronizing the diagonal legs.

Walk on flat ground for 8 s with GaitGenerator("trot"). The rhythm generator runs at 100 Hz (control_every 5), the PD at 500 Hz.
Three stages: (1) the standing controller from Chapter 6 as is (kp 40, target joint velocity 0) → (2) target joint velocity feedforward → (3) kp 80 + feedforward.
Metrics: mean speed (compared with the ideal L·f/β), joint tracking error, max torque, roll·pitch wobble, whether it fell.
Figures: out/ch07-fig05-trot-frames.png, out/ch07-fig06-trot-traces.png
Run: uv run python ch07-first-gait/03_trot.py   (--view --speed 0.5 recommended)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator
from quadbook.render import play, snapshots, track_camera, want_view
from quadbook.robot import load
from quadbook.sim import WALK_KP, rollout, standard_pd, summary

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CONTROL_EVERY = 5


def make(kp=WALK_KP, **kw):
    model, data = load(model_path())
    pd = standard_pd(model, data, kp=kp)
    gen = GaitGenerator("trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04, **kw)
    return model, data, pd, gen


print("trot 2 Hz, duty 0.5, step length 0.08 m, step height 0.04 m | ideal speed L·f/β = 0.320 m/s")
print("  stage                      speed[m/s] vs ideal  track err  max torque roll σ  pitch σ")
for name, kp, ff in (("(1) kp 40, velocity target 0", 40.0, False), ("(2) kp 40 + velocity feedforward", 40.0, True), ("(3) kp 80 + velocity feedforward", 80.0, True)):
    model, data, pd, gen = make(kp=kp)
    log = rollout(model, data, gen, pd, duration=8.0, control_every=CONTROL_EVERY, feedforward=ff)
    s = summary(log)
    print(f"  {name:26s}  {s['speed']:.3f}     {100*s['speed']/gen.ideal_speed:3.0f}%     {s['track_err']:4.1f}°    {s['tau_max']:5.1f} N·m  {s['roll_std']:.2f}°   {s['pitch_std']:.2f}°" + ("   fell" if s["fell"] else ""))
c = log["contact"][log["t"] >= 2.0]
print(f"  (3) distance traveled {log['x'][-1]-log['x'][0]:.2f} m, torso height {s['height']:.3f} m, measured foot contact ratio (LF RF LH RH) {np.round(c.mean(axis=0), 2)} (commanded duty 0.5)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(log["t"], log["x"] - log["x"][0], label="torso x (measured)")
axes[0].plot(log["t"], np.maximum(0, log["t"] - 0.5) * gen.ideal_speed, ls="--", color="#999999", label=f"ideal L f / beta = {gen.ideal_speed:.2f} m/s")
axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("distance [m]"); axes[0].set_title("(a) forward progress", fontsize=10); axes[0].legend(fontsize=8)
axes[1].plot(log["t"], np.degrees(log["pitch"]), label="pitch"); axes[1].plot(log["t"], np.degrees(log["roll"]), label="roll")
ax2 = axes[1].twinx(); ax2.plot(log["t"], log["z"], color="#2ca02c", lw=0.8, label="height"); ax2.set_ylabel("torso height [m]", color="#2ca02c")
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("angle [deg]"); axes[1].set_title("(b) torso attitude and height", fontsize=10); axes[1].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ch07-fig06-trot-traces.png", dpi=200); plt.close(fig)

# frames: one cycle (0.5 s) in four shots, camera tracking the torso
model, data, pd, gen = make()
cam = track_camera(model, distance=1.3, azimuth=140, elevation=-15)
dt_ctrl = model.opt.timestep * CONTROL_EVERY
state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
def _step():
    if data.time >= 0.5 and state["i"] % CONTROL_EVERY == 0:
        state["q"], state["qd"] = gen.targets_with_velocity(dt_ctrl)
    data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
print("saved", snapshots(model, data, (3.0, 3.125, 3.25, 3.375), OUT / "ch07-fig05-trot-frames.png", camera=cam, step_fn=_step, width=600, height=450))
print("saved", OUT / "ch07-fig06-trot-traces.png")

if want_view():
    model, data, pd, gen = make()
    state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    play(model, data, step_fn=_step, duration=10.0, title="trot 2 Hz (open-loop CPG + joint PD kp 80 + velocity feedforward)")
