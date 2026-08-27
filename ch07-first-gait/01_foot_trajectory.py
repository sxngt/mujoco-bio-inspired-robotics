# labs/ch07-first-gait/01_foot_trajectory.py
"""7.1 The simplest gait: designing a sinusoidal trajectory.

(a) What path does the foot trace when sine waves are fed straight into the joints (hip only / hip+knee with a phase offset)?
(b) Draw the foot path first (stance line + swing arc), then get joint angles by inverse kinematics.
(c) FK check: compare the foot position from our formula with the one MuJoCo computes.
Figures: out/ch07-fig01-foot-paths.png, out/ch07-fig02-swing-poses.png
Run: uv run python ch07-first-gait/01_foot_trajectory.py   (--view: the LF leg of a robot held in the air follows the trajectory)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import FootTrajectory, LegKinematics
from quadbook.render import play, poses, track_camera, want_view
from quadbook.robot import STAND_POSE_BALANCED, load

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
kin = LegKinematics()
HIP0, KNEE0 = STAND_POSE_BALANCED[1], STAND_POSE_BALANCED[2]
x0, z0 = kin.forward(HIP0, KNEE0)
phi = np.linspace(0, 1, 400, endpoint=False)

# ---- (a) joint sine waves ----
A = 0.25
hip_only = [(HIP0 + A * np.sin(2 * np.pi * p), KNEE0) for p in phi]
hip_knee = [(HIP0 + A * np.sin(2 * np.pi * p), KNEE0 + A * np.sin(2 * np.pi * p + np.pi / 2)) for p in phi]
path_a = np.array([kin.forward(h, k) for h, k in hip_only])
path_b = np.array([kin.forward(h, k) for h, k in hip_knee])
print(f"Foot position in the standing pose: x0 = {x0:+.4f}, z0 = {z0:+.4f} m (relative to the hip)")
print(f"(a) hip sine only (±{A} rad): foot height varies by {1000*(path_a[:,1].max()-path_a[:,1].min()):.0f} mm, lowest point is {1000*(z0-path_a[:,1].min()):.0f} mm below z0")
print(f"(b) hip+knee sine (90° phase offset): foot loop width {1000*(path_b[:,0].max()-path_b[:,0].min()):.0f} mm, height {1000*(path_b[:,1].max()-path_b[:,1].min()):.0f} mm, {1000*max(0, z0-path_b[:,1].min()):.0f} mm below the ground line")

# ---- (b) designed foot trajectory + IK ----
traj = FootTrajectory(x0, z0, duty=0.5, step_length=0.08, step_height=0.04)
path_c = np.array([traj(p) for p in phi])
q_c = np.array([kin.inverse(x, z) for x, z in path_c])
print(f"(c) designed trajectory: stance line {1000*0.08:.0f} mm, swing height {1000*0.04:.0f} mm | joint range hip {q_c[:,0].min():.2f}~{q_c[:,0].max():.2f}, knee {q_c[:,1].min():.2f}~{q_c[:,1].max():.2f} rad")

# ---- FK check ----
model, data = load(model_path())
errs = []
for h, k in q_c[::40]:
    data.qpos[7:19] = STAND_POSE_BALANCED; data.qpos[8], data.qpos[9] = h, k
    mujoco.mj_forward(model, data)
    rel = data.geom("LF_foot").xpos - data.body("LF_thigh").xpos
    errs.append(np.hypot(*(rel[[0, 2]] - kin.forward(h, k))))
print(f"FK check: max difference between our formula and the MuJoCo foot position {1e3*max(errs):.3f} mm")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), sharey=True)
for ax, path, title in zip(axes, (path_a, path_b, path_c), ("(a) hip sine only", "(b) hip + knee sine, 90 deg apart", "(c) designed: stance line + swing arc")):
    ax.plot(path[:, 0], path[:, 1], lw=1.5)
    ax.plot(path[0, 0], path[0, 1], "o", color="#d62728")
    ax.axhline(z0, color="#999999", ls=":", lw=1, label="ground line (stance height)")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.set_xlabel("foot x [m] (forward +)")
    ax.set_xlim(x0 - 0.1, x0 + 0.1); ax.set_xticks(np.round(np.arange(-0.12, 0.05, 0.04), 2))
axes[0].set_ylabel("foot z [m] (hip = 0)"); axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ch07-fig01-foot-paths.png", dpi=200); plt.close(fig)

# ---- pose frames: the LF leg of a robot held in the air at four points on the trajectory ----
qs, labels = [], []
for p in (0.0, 0.25, 0.5, 0.75):
    h, k = kin.inverse(*traj(p))
    q = np.concatenate([[0, 0, 0.45], [1, 0, 0, 0], STAND_POSE_BALANCED]); q[8], q[9] = h, k
    qs.append(q); labels.append(f"phase {p:.2f}: " + ("stance" if p < 0.5 else "swing"))
cam = track_camera(model, distance=1.05, azimuth=150, elevation=-14)
print("saved", poses(model, data, qs, OUT / "ch07-fig02-swing-poses.png", camera=cam, labels=labels, width=600, height=500))
print("saved", OUT / "ch07-fig01-foot-paths.png")

if want_view():
    model, data = load(model_path())
    model.opt.gravity[:] = 0                         # keep the robot floating and move only the leg
    data.qpos[2] = 0.45
    from quadbook.sim import standard_pd
    pd = standard_pd(model, data)
    def _step():
        h, k = kin.inverse(*traj((data.time * 0.5) % 1.0))   # slowly, at 0.5 Hz
        q = STAND_POSE_BALANCED.copy(); q[1], q[2] = h, k
        data.ctrl[:] = pd.torque(data, q); mujoco.mj_step(model, data)
    play(model, data, step_fn=_step, duration=8.0, title="In the air: the LF leg follows the designed trajectory (0.5 Hz)")
