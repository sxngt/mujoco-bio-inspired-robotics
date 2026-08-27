# labs/ch04-mujoco-first-steps/06_render_offscreen.py
"""Render to an image file without a window (for servers, notebooks, and book figures).

mujoco.Renderer returns the scene seen by a camera as a numpy array. It captures the scene
from 02_viewer.py at three moments, 0.1 s, 0.4 s, and 0.8 s, and saves them to out/ch04-fig02-first-scene.png.
Run: uv run python ch04-mujoco-first-steps/06_render_offscreen.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

import mujoco

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

XML = """
<mujoco model="render_demo">
  <option timestep="0.002"/>
  <visual><global offwidth="1600" offheight="1000"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.65 0.65 0.65" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/>
  </asset>
  <worldbody>
    <light pos="1 1 3" dir="-0.3 -0.3 -1" diffuse="0.9 0.9 0.9" castshadow="true"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
    <body name="box1" pos="0 0 0.6"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1" rgba="0.2 0.5 0.9 1"/></body>
    <body name="box2" pos="0.05 0.02 1.1"><freejoint/><geom type="box" size="0.08 0.08 0.08" mass="0.5" rgba="0.9 0.4 0.2 1"/></body>
    <body name="ball" pos="-0.3 0.2 0.8"><freejoint/><geom type="sphere" size="0.07" mass="0.3" rgba="0.3 0.8 0.3 1"/></body>
    <camera name="side" pos="1.9 -1.9 1.35" xyaxes="0.7 0.7 0 -0.35 0.35 0.87"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=1000, width=1600)

frames = []
mujoco.mj_forward(model, data)          # compute derived quantities (world coordinates etc.) before the first frame
for t_snap in (0.1, 0.4, 0.8):
    while data.time < t_snap:
        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="side")
    frames.append(renderer.render().copy())

# Stitch the three moments side by side
strip = np.concatenate(frames, axis=1)
Image.fromarray(strip).save(OUT / "ch04-fig02-first-scene.png")
print("saved", OUT / "ch04-fig02-first-scene.png", strip.shape)
renderer.close()
