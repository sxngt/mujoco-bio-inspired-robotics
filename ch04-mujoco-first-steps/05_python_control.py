# labs/ch04-mujoco-first-steps/05_python_control.py
"""Driving the simulation from Python: send one joint to a target angle.

The same simple pendulum is controlled two ways.
(A) <motor> actuator + PD control written directly in Python: each step torque = kp*(target-angle) - kd*angular velocity
(B) <position> actuator: MuJoCo computes the same PD internally. Only the target angle goes into ctrl.
Both responses are plotted in one figure: out/ch04-fig07-control-response.png
Run: uv run python ch04-mujoco-first-steps/05_python_control.py
Add --view to watch the same scene in real time in a viewer window (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # draw to file only, no display (also works in the mjpython worker thread)
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook.render import play, snapshots, want_view

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

KP, KD = 20.0, 1.0
TARGET = np.deg2rad(60)

XML = """
<mujoco model="pendulum_control">
  <option timestep="0.002"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="0 -2.0 0.7" xyaxes="1 0 0 0 0 1"/>
    <!-- Stand: a post rising from the floor to the pivot plus a crossbar. Visual only, no collisions (contype 0) -->
    <geom type="cylinder" fromto="0 0.15 0 0 0.15 1" size="0.015" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0.15 1 0 0 1" size="0.012" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="sphere" pos="0 0 1" size="0.03" rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/>
    <body pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    {actuator}
  </actuator>
</mujoco>
"""
MOTOR = '<motor name="act" joint="hinge" gear="1" ctrlrange="-20 20"/>'
POSITION = f'<position name="act" joint="hinge" kp="{KP}" kv="{KD}" ctrlrange="-3.14 3.14"/>'


def simulate(actuator_xml, controller, duration=3.0):
    model = mujoco.MjModel.from_xml_string(XML.format(actuator=actuator_xml))
    data = mujoco.MjData(model)
    t, q = [], []
    while data.time < duration:
        data.ctrl[0] = controller(data)      # compute the control input every step
        mujoco.mj_step(model, data)
        t.append(data.time)
        q.append(data.qpos[0])
    return np.array(t), np.array(q)


# (A) hand-written PD: compute the torque directly and feed it to the motor
def pd_torque(data):
    err = TARGET - data.qpos[0]
    return KP * err - KD * data.qvel[0]


# (B) position actuator: only the target angle goes in
def target_only(data):
    return TARGET


tA, qA = simulate(MOTOR, pd_torque)
tB, qB = simulate(POSITION, target_only)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(tA, np.rad2deg(qA), label="(A) motor + PD written in Python")
ax.plot(tB, np.rad2deg(qB), "--", label="(B) position actuator (built-in PD)")
ax.axhline(np.rad2deg(TARGET), color="#999999", lw=1, ls=":", label="target 60 deg")
ax.set_xlabel("time [s]")
ax.set_ylabel("joint angle [deg]")
ax.set_title(f"pendulum joint, kp={KP}, kd={KD}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ch04-fig07-control-response.png", dpi=200)
plt.close(fig)

# Capture four moments of the motion under method (A)
_m = mujoco.MjModel.from_xml_string(XML.format(actuator=MOTOR)); _d = mujoco.MjData(_m)
def _step():
    _d.ctrl[0] = pd_torque(_d); mujoco.mj_step(_m, _d)
print("saved", snapshots(_m, _d, (0.0, 0.15, 0.4, 3.0), OUT / "ch04-fig06-pendulum-control.png", camera="front", step_fn=_step, width=600, height=600))

print(f"(A) angle after 3 s: {np.rad2deg(qA[-1]):6.2f} deg  | (B) angle after 3 s: {np.rad2deg(qB[-1]):6.2f} deg  (target 60 deg)")
print("max difference between the two curves:", f"{np.rad2deg(np.abs(qA - qB)).max():.3f} deg")

if want_view():
    _m = mujoco.MjModel.from_xml_string(XML.format(actuator=MOTOR)); _d = mujoco.MjData(_m)
    def _step():
        _d.ctrl[0] = pd_torque(_d); mujoco.mj_step(_m, _d)
    play(_m, _d, step_fn=_step, duration=5.0, title="Simple pendulum PD control (target 60 deg)")
