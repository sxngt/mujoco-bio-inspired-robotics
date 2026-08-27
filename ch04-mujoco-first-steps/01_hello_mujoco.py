# labs/ch04-mujoco-first-steps/01_hello_mujoco.py
"""First MuJoCo run: drop a single box onto the floor.

Build the model from an XML string (mjModel), prepare the state container (mjData),
then call mj_step repeatedly to simulate one second. No window is opened.
Run: uv run python ch04-mujoco-first-steps/01_hello_mujoco.py
Add --view to watch the same scene in real time in a viewer window (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import mujoco
from quadbook.render import play, snapshots, want_view

XML = """
<mujoco model="hello">
  <option timestep="0.002"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="side" pos="2.2 -2.2 1.1" xyaxes="0.7 0.7 0 -0.25 0.25 0.94"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="box" pos="0 0 1.0">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1" rgba="0.2 0.5 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)   # constants: the model description
data = mujoco.MjData(model)                    # variables: the state that changes over time

print(f"MuJoCo {mujoco.__version__}")
print(f"nq={model.nq} (number of position coordinates), nv={model.nv} (number of velocity coordinates), "
      f"nbody={model.nbody}, ngeom={model.ngeom}, timestep={model.opt.timestep}")

# 1 second = 500 steps. Every 0.1 s, print the box height and the number of contacts.
steps_per_report = int(0.1 / model.opt.timestep)
for i in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    if (i + 1) % steps_per_report == 0:
        z = data.body("box").xpos[2]
        print(f"t={data.time:4.2f}s  box z={z:6.3f} m  contacts={data.ncon}")

# Replay the same scene from the start and save three moments as a figure
data = mujoco.MjData(model)
out = snapshots(model, data, (0.05, 0.3, 0.6), Path(__file__).resolve().parent / "out" / "ch04-fig01-hello-drop.png", camera="side")
print("saved", out)

if want_view():
    play(model, mujoco.MjData(model), duration=3.0, title="Falling box")
