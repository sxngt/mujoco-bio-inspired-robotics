# labs/ch07-first-gait/06_open_loop_limits.py
"""7.6 Why our gait is stiff: witnessing the limits of open-loop control.

The generator never reads the robot state. We watch the consequences in four situations.
(1) Even on flat ground: commanded foot height and actual foot contact disagree (feet touch down earlier or lift off later than planned).
(2) Side push: the robot withstood 40 N while standing; how much does it take to topple it while walking?
(3) Uphill: tilt the gravity vector to make 5° and 10° slopes.
(4) Payload: add 2 kg and 6 kg to the torso. Friction: lower the foot-ground friction coefficient from 1.0 to 0.4 and 0.15.
Figures: out/ch07-fig11-push-fall-frames.png, out/ch07-fig12-open-loop-traces.png
Run: uv run python ch07-first-gait/06_open_loop_limits.py   (--view --speed 0.5: pushes while walking, 20 N at 3 s, 40 N at 6 s, 60 N at 9 s)
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
from quadbook.robot import load, reset_stand
from quadbook.sim import WALK_KP, rollout, standard_pd, summary

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
TROT = dict(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04)
CE = 5


def setup(slope_deg=0.0, payload=0.0, friction=None):
    model, data = load(model_path())
    if slope_deg:
        a = np.radians(slope_deg)                       # uphill: gravity tilts backward (−x)
        model.opt.gravity[:] = [-9.81 * np.sin(a), 0.0, -9.81 * np.cos(a)]
    if payload:
        model.body_mass[model.body("torso").id] += payload
        mujoco.mj_setConst(model, data)             # recompute mass-derived quantities. Caution: data reverts to qpos0 (legs extended)
        reset_stand(model, data)                    # so re-initialize from the standing keyframe
    if friction is not None:
        model.geom_friction[:, 0] = friction            # a contact pair uses the larger of the two geom frictions, so lower both foot and ground
    mujoco.mj_forward(model, data)
    return model, data, standard_pd(model, data, kp=WALK_KP), GaitGenerator(**TROT)


def pusher(F, t0=3.0, dur=0.2):
    def fn(model, data):
        torso = model.body("torso").id
        data.xfrc_applied[torso, :] = 0.0
        if t0 <= data.time < t0 + dur:
            data.xfrc_applied[torso, 1] = F
    return fn


# ---- (1) flat ground: commanded vs actual contact ----
print("(1) trot on flat ground, LF leg over 2 s: mismatch between commanded stance and actual contact")
slow = None
for f in (2.0, 1.0):
    model, data, pd, gen = setup(); gen = GaitGenerator(**{**TROT, "freq": f})
    log = rollout(model, data, gen, pd, duration=6.0, control_every=CE)
    m = (log["t"] >= 3.0) & (log["t"] < 5.0)
    cmd_stance = log["foot_cmd"][m][:, 0] < 1e-6
    act_stance = log["contact"][m][:, 0].astype(bool)
    early = (act_stance & ~cmd_stance).mean(); late = (~act_stance & cmd_stance).mean()
    print(f"    {f:.0f} Hz: commanded stance {cmd_stance.mean():.2f}, actual contact {act_stance.mean():.2f} | touched down earlier than planned {early:.2f}, in the air during planned stance {late:.2f} | speed {summary(log)['speed']:.3f} m/s ({100*summary(log)['speed']/gen.ideal_speed:.0f}%)")
    if f == 2.0: base = log
    else: slow = log
s0 = summary(base)

# ---- (2) push ----
print("(2) 0.2 s side push while walking (limit while standing: 40 N)")
push_logs = {}
for F in (20, 40, 60, 80, 100):
    model, data, pd, gen = setup()
    log = rollout(model, data, gen, pd, duration=6.0, control_every=CE, on_step=pusher(F))
    push_logs[F] = log
    print(f"    {F:3d} N: fell {summary(log)['fell']}, y drift after the push {log['y'][-1]-log['y'][0]:+.2f} m")

# ---- (3)(4) slope, payload, friction ----
print("(3)(4) changing the environment (same generator, same gains)")
cases = {"flat": {}, "slope 5 deg": dict(slope_deg=5), "slope 10 deg": dict(slope_deg=10), "payload +2 kg": dict(payload=2.0), "payload +6 kg": dict(payload=6.0), "friction 0.4": dict(friction=0.4), "friction 0.15": dict(friction=0.15)}
logs = {}
for name, kw in cases.items():
    model, data, pd, gen = setup(**kw)
    log = rollout(model, data, gen, pd, duration=6.0, control_every=CE); logs[name] = log; s = summary(log)
    print(f"    {name:14s}: speed {s['speed']:+.3f} m/s ({100*s['speed']/gen.ideal_speed:+.0f}% of ideal), height {s['height']:.3f} m, pitch σ {s['pitch_std']:.2f}°, fell {s['fell']}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
t = slow["t"]; mm = (t >= 3.0) & (t < 5.0)
axes[0].plot(t[mm], 1000 * slow["foot_cmd"][mm][:, 0], label="commanded foot height [mm]")
axes[0].fill_between(t[mm], 0, 50, where=slow["contact"][mm][:, 0].astype(bool), color="#2ca02c", alpha=0.25, label="actual contact (LF)")
axes[0].set_xlabel("time [s]"); axes[0].set_title("(a) LF at 1 Hz: planned vs actual contact", fontsize=10); axes[0].legend(fontsize=8)
for F, log in push_logs.items():
    axes[1].plot(log["t"], log["y"] - log["y"][0], lw=0.9, label=f"{F} N push at 3 s")
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("sideways drift y [m]"); axes[1].set_title("(b) side push while trotting: no return to path", fontsize=10); axes[1].legend(fontsize=8)
for name, log in logs.items():
    axes[2].plot(log["t"], log["x"] - log["x"][0], lw=0.9, label=name)
axes[2].set_xlabel("time [s]"); axes[2].set_ylabel("distance [m]"); axes[2].set_title("(c) progress in changed environments", fontsize=10); axes[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch07-fig12-open-loop-traces.png", dpi=200); plt.close(fig)

# frames: after the push
model, data, pd, gen = setup()
cam = track_camera(model, distance=1.4, azimuth=90, elevation=-12)
push = pusher(60); dt_ctrl = model.opt.timestep * CE
state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
def _step():
    if data.time >= 0.5 and state["i"] % CE == 0: state["q"], state["qd"] = gen.targets_with_velocity(dt_ctrl)
    data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); push(model, data); mujoco.mj_step(model, data); state["i"] += 1
print("saved", snapshots(model, data, (2.9, 3.2, 3.6, 5.0), OUT / "ch07-fig11-push-fall-frames.png", camera=cam, step_fn=_step, width=600, height=450))
print("saved", OUT / "ch07-fig12-open-loop-traces.png")

if want_view():
    # Push the walking robot harder every 3 s: 20 N at 3 s (shoved but keeps walking), 40 N at 6 s (shoved hard but stays up), 60 N at 9 s (falls).
    # Each push lasts only 0.2 s and is easy to miss, so it is narrated in the terminal. The push is the moment the torso jumps sideways.
    from quadbook.sim import fallen
    model, data, pd, gen = setup(); state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    torso = model.body("torso").id
    schedule = {3.0: 20.0, 6.0: 40.0, 9.0: 60.0}
    said = set()
    def _step_view():
        if data.time >= 0.5 and state["i"] % CE == 0: state["q"], state["qd"] = gen.targets_with_velocity(dt_ctrl)
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"])
        data.xfrc_applied[torso, :] = 0.0
        for t0, F in schedule.items():
            if t0 <= data.time < t0 + 0.2:
                data.xfrc_applied[torso, 1] = F
                if t0 not in said:
                    said.add(t0); print(f"[{data.time:4.1f} s] push from the side (+y): {F:.0f} N for 0.2 s starts")
            if (t0 + 1.0) <= data.time < t0 + 1.0 + model.opt.timestep and (t0, "r") not in said:
                said.add((t0, "r")); print(f"[{data.time:4.1f} s] 1 s after the push: " + ("fell" if fallen(data) else f"stayed up, shoved {data.body('torso').xpos[1]:+.2f} m sideways (does not return to the original path)"))
        mujoco.mj_step(model, data); state["i"] += 1
    print("[viewer] 20 N at 3 s → 40 N at 6 s → 60 N at 9 s. The push is the moment the torso jumps sideways. Ends at 12 s.")
    play(model, data, step_fn=_step_view, duration=12.0, title="side pushes while trotting: 20 N (3 s), 40 N (6 s), 60 N (9 s)")
