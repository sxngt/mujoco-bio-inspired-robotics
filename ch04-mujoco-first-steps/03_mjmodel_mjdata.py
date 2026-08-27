# labs/ch04-mujoco-first-steps/03_mjmodel_mjdata.py
"""See the division of labor between mjModel and mjData with your own eyes.

Build a two-joint arm (shoulder hinge + elbow hinge), then:
- print what lives in mjModel (constants) and what lives in mjData (variables).
- set qpos by hand and call mj_forward to watch xpos (world coordinates) follow.
- compare mj_forward (computation only) with mj_step (computation + time advance).
Run: uv run python ch04-mujoco-first-steps/03_mjmodel_mjdata.py
Add --view to watch the same scene in real time in a viewer window (macOS: uv run mjpython ... --view).
"""

import numpy as np

from pathlib import Path

import mujoco
from quadbook.render import play, poses, want_view

XML = """
<mujoco model="two_link_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="front" pos="0.3 -1.6 0.6" xyaxes="1 0 0 0 0.2 1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.5">
      <geom type="cylinder" size="0.05 0.02" rgba="0.3 0.3 0.3 1"/>
      <body name="upper" pos="0 0 0">
        <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom name="upper_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
        <body name="lower" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" damping="0.5"/>
          <geom name="lower_geom" type="capsule" fromto="0 0 0 0.25 0 0" size="0.025" mass="0.6"/>
          <site name="tip" pos="0.25 0 0" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1" ctrlrange="-5 5"/>
    <motor name="elbow_motor" joint="elbow" gear="1" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print("=== mjModel: compiled constants ===")
print(f"nq={model.nq}, nv={model.nv}, nu={model.nu} (number of actuators), nbody={model.nbody}, njnt={model.njnt}")
print("body names:", [model.body(i).name for i in range(model.nbody)])
print("joint names:", [model.joint(i).name for i in range(model.njnt)])
print("timestep:", model.opt.timestep, "| gravity:", model.opt.gravity)
print("upper_geom mass:", model.body("upper").mass[0], "kg | lower mass:", model.body("lower").mass[0], "kg")

print("\n=== mjData: state that changes over time ===")
print("initial qpos:", data.qpos, "| qvel:", data.qvel, "| time:", data.time)
print("tip world position (not computed yet):", data.site("tip").xpos)

# Set qpos by hand and call mj_forward: time does not advance, only derived quantities (world coordinates etc.) are computed
data.qpos[:] = [np.deg2rad(45), np.deg2rad(-60)]
mujoco.mj_forward(model, data)
print("qpos set to (45°, -60°) then mj_forward →  tip:", np.round(data.site("tip").xpos, 3), "| time:", data.time)

# mj_step: compute forces and advance time by one step. The arm starts to sag under gravity.
for _ in range(500):          # 1 second
    mujoco.mj_step(model, data)
print("qpos(deg) after 1 s of free fall:", np.round(np.rad2deg(data.qpos), 1), "| time:", round(data.time, 3))

# Apply a force through the actuator: ctrl lives in mjData (it is state, not model)
data.ctrl[:] = [3.0, 0.0]     # 3 N·m at the shoulder
for _ in range(500):
    mujoco.mj_step(model, data)
print("qpos(deg) after 1 s of 3 N·m shoulder torque:", np.round(np.rad2deg(data.qpos), 1))

# The same mjModel can spawn a second world: the freedom that comes from separating constants and state
data2 = mujoco.MjData(model)
print("\nsame model, new data2 time:", data2.time, "| existing data time:", round(data.time, 3))

# Three poses as a figure: initial (0, 0), hand-set (45°, -60°), and the pose after applying torque
final_q = data.qpos.copy()
out = poses(model, data2, [[0, 0], [np.deg2rad(45), np.deg2rad(-60)], final_q],
            Path(__file__).resolve().parent / "out" / "ch04-fig03-two-link-poses.png", camera="front",
            labels=["qpos = (0, 0)", "qpos = (45, -60) deg, mj_forward", "after 3 N.m for 1 s"])
print("saved", out)

if want_view():
    d = mujoco.MjData(model)
    d.qpos[:] = [np.deg2rad(45), np.deg2rad(-60)]
    def _step():
        d.ctrl[:] = [3.0, 0.0] if d.time >= 1.0 else [0.0, 0.0]   # 3 N·m at the shoulder after 1 s
        mujoco.mj_step(model, d)
    play(model, d, step_fn=_step, duration=4.0, title="Two-joint arm: shoulder torque after 1 s")
