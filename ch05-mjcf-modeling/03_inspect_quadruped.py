# labs/ch05-mjcf-modeling/03_inspect_quadruped.py
"""Looks inside the finished robot (models/quadruped.xml) and saves a picture of the standing pose.

Output figure: out/ch05-fig04-quadruped-stand.png
Run: uv run python ch05-mjcf-modeling/03_inspect_quadruped.py
Add --view to watch the same scene live in the viewer window (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import numpy as np
from PIL import Image

import mujoco
from quadbook import model_path
from quadbook.render import play, want_view
from quadbook.robot import JOINT_NAMES, STAND_POSE, load, pd_torque

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

model, data = load(model_path())

print(f"nbody={model.nbody}, njnt={model.njnt}, nq={model.nq}, nv={model.nv}, nu={model.nu}, nsensor={model.nsensor}")
print("total mass:", round(float(model.body_subtreemass[model.body('torso').id]), 3), "kg")
print("joint order:", JOINT_NAMES)
print("joint ranges (rad):")
for name in JOINT_NAMES[:3]:
    print(f"  {name:14s}", np.round(model.joint(name).range, 2))
print("actuator torque limit:", model.actuator("LF_hip").ctrlrange, "N·m")
print("sensor data length:", model.nsensordata, "| e.g. torso_quat =", np.round(data.sensor("torso_quat").data, 3))

# Torso height and foot positions in the standing pose (keyframe)
print("torso height:", round(data.body("torso").xpos[2], 3), "m")
for leg in ("LF", "RF", "LH", "RH"):
    p = data.geom(f"{leg}_foot").xpos
    print(f"  {leg} foot: x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}")

# Left-right symmetry check: left leg mass sum == right leg mass sum
def subtree_mass(body):
    return float(model.body_subtreemass[model.body(body).id])
print("left vs right mass:", subtree_mass("LF_hip") + subtree_mass("LH_hip"), "vs", subtree_mass("RF_hip") + subtree_mass("RH_hip"))

# Capture the standing pose from two cameras
renderer = mujoco.Renderer(model, height=1000, width=1600)
frames = []
for cam in ("side", "front"):
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())
Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch05-fig04-quadruped-stand.png")
renderer.close()
print("saved", OUT / "ch05-fig04-quadruped-stand.png")

if want_view():
    # Open the window while PD holds the standing pose. Try grabbing and dragging the robot with Cmd (Mac)/Ctrl+drag
    m2, d2 = load(model_path())
    def _step():
        d2.ctrl[:] = pd_torque(d2, STAND_POSE); mujoco.mj_step(m2, d2)
    play(m2, d2, step_fn=_step, title="Finished robot: standing pose (drag to push it)")
