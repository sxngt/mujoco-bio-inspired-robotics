# labs/ch05-mjcf-modeling/02_one_leg.py
"""Model v1: a single leg. Mounts three joints (abduction, hip, knee) on a test stand fixed to the world,
sweeps the joint angles, and plots the region the foot can reach (the workspace).

Output figure: out/ch05-fig03-leg-workspace.png
Run: uv run python ch05-mjcf-modeling/02_one_leg.py
Add --view to watch the same scene live in the viewer window (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # Draw to files only, no display (also works in the mjpython worker thread)
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook.render import play, poses, want_view

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

XML = """
<mujoco model="v1_one_leg">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <joint damping="0.2" armature="0.01"/>
    <default class="leg"><geom type="capsule" rgba="0.55 0.55 0.6 1"/></default>
  </default>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="1 1 3" dir="-0.3 -0.3 -1"/>
    <camera name="side" pos="0.75 -0.75 0.3" xyaxes="0.7 0.7 0 0 0 1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <!-- Test stand: the hip is fixed to the world (no freejoint) -->
    <body name="hip_mount" pos="0 0 0.5">
      <geom type="box" size="0.03 0.03 0.03" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <body name="hip" pos="0 0 0">
        <joint name="abduction" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom class="leg" fromto="0 0 0 0 0.04 0" size="0.03" mass="0.2"/>
        <body name="thigh" pos="0 0.04 0">
          <joint name="hip" type="hinge" axis="0 1 0" range="-1.2 2.4"/>
          <geom class="leg" fromto="0 0 0 0 0 -0.2" size="0.025" mass="0.6"/>
          <body name="calf" pos="0 0 -0.2">
            <joint name="knee" type="hinge" axis="0 1 0" range="-2.6 -0.3"/>
            <geom class="leg" fromto="0 0 0 0 0 -0.2" size="0.018" mass="0.3"/>
            <geom name="foot" type="sphere" pos="0 0 -0.2" size="0.02" mass="0.05" rgba="0.15 0.15 0.15 1"/>
            <site name="foot_site" pos="0 0 -0.2" size="0.01"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
print(f"njnt={model.njnt}, nq={model.nq}, nv={model.nv} (no freejoint, so they equal the joint count)")
for j in ("abduction", "hip", "knee"):
    print(f"  {j:9s} range = {np.round(model.joint(j).range, 2)} rad")

# Sweep hip and knee angles and record the foot (x, z) position (abduction = 0)
hip_range = np.linspace(*model.joint("hip").range, 60)
knee_range = np.linspace(*model.joint("knee").range, 60)
pts = []
for h in hip_range:
    for k in knee_range:
        data.qpos[:] = [0.0, h, k]
        mujoco.mj_forward(model, data)
        p = data.site("foot_site").xpos - data.body("hip_mount").xpos
        pts.append((p[0], p[2]))
pts = np.array(pts)

# Foot position in the standing pose (0.8, -1.5)
data.qpos[:] = [0.0, 0.8, -1.5]
mujoco.mj_forward(model, data)
stand = data.site("foot_site").xpos - data.body("hip_mount").xpos
print(f"foot in standing pose (hip 0.8, knee -1.5): x={stand[0]:+.3f} m, z={stand[2]:+.3f} m (relative to hip)")

# Render three poses: fully extended, standing, and knee deeply folded
print("saved", poses(model, data, [[0, 0, -0.3], [0, 0.8, -1.5], [0.5, 1.6, -2.4]], OUT / "ch05-fig02-one-leg-poses.png",
                      camera="side", labels=["(0, 0, -0.3)", "standing (0, 0.8, -1.5)", "(0.5, 1.6, -2.4)"], width=600, height=600))

fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.scatter(pts[:, 0], pts[:, 1], s=4, color="#9ecae1", label="reachable foot positions")
ax.plot(0, 0, "ks", ms=8, label="hip joint")
ax.plot(stand[0], stand[2], "ro", ms=9, label="standing pose (0.8, -1.5)")
ax.add_patch(plt.Circle((0, 0), 0.4, fill=False, ls="--", color="#999999"))
ax.set_aspect("equal")
ax.set_xlabel("x forward [m]")
ax.set_ylabel("z up [m]")
ax.set_title("foot workspace, thigh 0.2 m + calf 0.2 m")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "ch05-fig03-leg-workspace.png", dpi=200)
print("saved", OUT / "ch05-fig03-leg-workspace.png")

if want_view():
    # Kinematics only, no physics: slowly sweep hip, knee, and abduction with sine waves to see the workspace
    d = mujoco.MjData(model)
    def _step():
        d.time += model.opt.timestep
        t = d.time
        d.qpos[:] = [0.4 * np.sin(0.5 * t), 0.8 + 0.8 * np.sin(t), -1.5 + 0.9 * np.sin(1.3 * t)]
        mujoco.mj_forward(model, d)
    play(model, d, step_fn=_step, duration=20.0, title="One leg: joint sweep (kinematics only)")
