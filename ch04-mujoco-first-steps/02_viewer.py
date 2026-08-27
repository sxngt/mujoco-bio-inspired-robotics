# labs/ch04-mujoco-first-steps/02_viewer.py
"""Meeting the world through the viewer: drop several boxes and open a window.

On macOS, launch_passive requires running with mjpython instead of python.
    uv run mjpython ch04-mujoco-first-steps/02_viewer.py
Linux/Windows:
    uv run python ch04-mujoco-first-steps/02_viewer.py
You can set a playback speed such as --speed 0.25 (1/4 slow motion).
Inside the window, drag with the mouse to rotate the view, double-click to select an object,
and Ctrl(Cmd on Mac)+drag to grab and pull an object. Space pauses.
"""

import mujoco
from quadbook.render import play

XML = """
<mujoco model="viewer_demo">
  <option timestep="0.002"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.65 0.65 0.65" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
    <body name="box1" pos="0 0 0.6"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1" rgba="0.2 0.5 0.9 1"/></body>
    <body name="box2" pos="0.05 0.02 1.1"><freejoint/><geom type="box" size="0.08 0.08 0.08" mass="0.5" rgba="0.9 0.4 0.2 1"/></body>
    <body name="ball" pos="-0.3 0.2 0.8"><freejoint/><geom type="sphere" size="0.07" mass="0.3" rgba="0.3 0.8 0.3 1"/></body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# Play back in real time, synced to the wall clock. Add --speed 0.25 to watch 4x slower.
play(model, data, duration=30.0, title="Two boxes and a ball")
