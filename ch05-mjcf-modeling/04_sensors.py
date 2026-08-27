# labs/ch05-mjcf-modeling/04_sensors.py
"""Reading sensors: drop the robot from a slight height and see what the IMU and foot touch sensors report.
The joints are held in the standing pose by PD (the controller itself is explained in Chapter 6).

Output figure: out/ch05-fig06-sensors-drop.png
Run: uv run python ch05-mjcf-modeling/04_sensors.py
Add --view to watch the same scene live in the viewer window (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # Draw to files only, no display (also works in the mjpython worker thread)
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.render import play, snapshots, want_view
from quadbook.robot import STAND_POSE, TOUCH_SENSORS, load, pd_torque

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

model, data = load(model_path())
data.qpos[2] = 0.5                      # Lift above the standing height (0.31) and drop
mujoco.mj_forward(model, data)

t, height, acc_z, touch = [], [], [], []
while data.time < 1.5:
    data.ctrl[:] = pd_torque(data, STAND_POSE)
    mujoco.mj_step(model, data)
    t.append(data.time)
    height.append(data.body("torso").xpos[2])
    acc_z.append(data.sensor("torso_acc").data[2])
    touch.append([data.sensor(n).data[0] for n in TOUCH_SENSORS])
t, height, acc_z, touch = map(np.array, (t, height, acc_z, touch))

land = t[np.argmax(touch.sum(axis=1) > 0)]
print(f"landing time {land:.3f} s | peak vertical acceleration right after landing {acc_z.max():.1f} m/s² | torso height at rest {height[-1]:.3f} m")
print("foot contact forces at rest (N):", np.round(touch[-1], 1), "| total", round(float(touch[-1].sum()), 1), "N ≈ body weight", round(float(model.body_subtreemass[model.body('torso').id] * 9.81), 1), "N")
q = data.sensor("torso_quat").data
print("torso orientation quaternion at rest (w x y z):", np.round(q, 3), "→ nearly (1,0,0,0) = level")

# Four moments of the drop
_m, _d = load(model_path()); _d.qpos[2] = 0.5
def _step():
    _d.ctrl[:] = pd_torque(_d, STAND_POSE); mujoco.mj_step(_m, _d)
print("saved", snapshots(_m, _d, (0.0, 0.2, 0.3, 1.5), OUT / "ch05-fig05-drop-frames.png", camera="side", step_fn=_step, width=700, height=500))

fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
axes[0].plot(t, height); axes[0].set_ylabel("torso z [m]")
axes[1].plot(t, acc_z); axes[1].set_ylabel("IMU acc z [m/s²]")
for k, n in enumerate(TOUCH_SENSORS):
    axes[2].plot(t, touch[:, k], label=n)
axes[2].set_ylabel("foot touch [N]"); axes[2].set_xlabel("time [s]"); axes[2].legend(fontsize=8, ncol=4)
for ax in axes:
    ax.axvline(land, color="#999999", ls=":", lw=1)
axes[0].set_title("drop from 0.5 m with PD holding the standing pose")
fig.tight_layout()
fig.savefig(OUT / "ch05-fig06-sensors-drop.png", dpi=200)
print("saved", OUT / "ch05-fig06-sensors-drop.png")

if want_view():
    m2, d2 = load(model_path()); d2.qpos[2] = 0.5
    def _step():
        d2.ctrl[:] = pd_torque(d2, STAND_POSE); mujoco.mj_step(m2, d2)
    play(m2, d2, step_fn=_step, duration=5.0, title="0.5 m drop (PD holds the pose)")
