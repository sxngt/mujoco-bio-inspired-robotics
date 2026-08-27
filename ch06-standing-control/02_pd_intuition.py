# labs/ch06-standing-control/02_pd_intuition.py
"""6.2 The principle of PD control, checked on a single joint.

Send the same simple pendulum (Chapter 4) to 60 degrees with three controllers.
  P only                    : torque = kp·(target − angle)                          → a spring alone, so it keeps oscillating
  PD                        : torque = kp·(target − angle) − kd·angular velocity    → the damper absorbs the oscillation, but gravity keeps it short of the target
  PD + gravity compensation : the torque above + gravity torque (qfrc_bias)         → the error disappears
Then compute the natural frequency ω = sqrt(kp/I) and the damping ratio ζ = kd/(2·sqrt(kp·I)) from the joint inertia I and compare them with the response.
Figures: out/ch06-fig04-pd-terms.png, out/ch06-fig05-pd-frames.png
Run: uv run python ch06-standing-control/02_pd_intuition.py
Add --view to watch the PD + gravity compensation controller in the viewer.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook.render import play, snapshots, want_view

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

XML = """
<mujoco model="pendulum_pd">
  <option timestep="0.002" integrator="implicitfast"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="0 -2.0 0.7" xyaxes="1 0 0 0 0 1"/>
    <geom type="cylinder" fromto="0 0.15 0 0 0.15 1" size="0.015" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0.15 1 0 0 1" size="0.012" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="sphere" pos="0 0 1" size="0.03" rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/>
    <body pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.0" armature="0.0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor name="act" joint="hinge" ctrlrange="-30 30"/></actuator>
</mujoco>
"""
TARGET = np.deg2rad(60)
KP, KD = 20.0, 1.0


def simulate(controller, duration=3.0):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    t, q = [], []
    while data.time < duration:
        data.ctrl[0] = controller(data)
        mujoco.mj_step(model, data)
        t.append(data.time); q.append(data.qpos[0])
    return model, data, np.array(t), np.array(q)


def p_only(data):
    return KP * (TARGET - data.qpos[0])


def pd(data):
    return KP * (TARGET - data.qpos[0]) - KD * data.qvel[0]


def pd_gravity(data):
    return KP * (TARGET - data.qpos[0]) - KD * data.qvel[0] + data.qfrc_bias[0]   # gravity compensation: the bias force computed by MuJoCo


runs = {"P only": p_only, "PD": pd, "PD + gravity comp": pd_gravity}
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for name, ctrl in runs.items():
    model, data, t, q = simulate(ctrl)
    ax.plot(t, np.degrees(q), label=name)
    print(f"{name:18s}: after 3 s {np.degrees(q[-1]):6.2f}° (target 60°), max {np.degrees(q.max()):6.2f}°")
ax.axhline(60, color="#999999", ls=":", lw=1)
ax.set_xlabel("time [s]"); ax.set_ylabel("angle [deg]"); ax.set_title(f"kp = {KP}, kd = {KD}: what each term does"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch06-fig04-pd-terms.png", dpi=200); plt.close(fig)

# ---- Second-order prediction: ω, ζ from the joint inertia I ----
model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
M = np.zeros((1, 1)); mujoco.mj_fullM(model, data, M); I = M[0, 0]
omega = np.sqrt(KP / I); zeta = KD / (2 * np.sqrt(KP * I))
period = 2 * np.pi / (omega * np.sqrt(1 - zeta**2))
overshoot = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) * 100
print(f"joint inertia I = {I:.4f} kg·m² (rod m l²/3 = {1*0.5**2/3:.4f}) | ω = {omega:.2f} rad/s | ζ = {zeta:.2f} | damped oscillation period {period:.2f} s | predicted overshoot {overshoot:.0f}%")
# Measurement: first peak of the PD + gravity compensation response
model, data, t, q = simulate(pd_gravity)
peak_i = np.argmax(q); print(f"measured: first peak {np.degrees(q[peak_i]):.1f}° at {t[peak_i]:.2f} s → overshoot {(q[peak_i]-TARGET)/TARGET*100:.0f}%")
kd_crit = 2 * np.sqrt(KP * I)
print(f"critical damping kd = 2·sqrt(kp·I) = {kd_crit:.2f} (the current kd {KD} is underdamped, ζ {zeta:.2f})")

model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model)
def _step():
    data.ctrl[0] = pd_gravity(data); mujoco.mj_step(model, data)
print("saved", snapshots(model, data, (0.0, 0.2, 0.5, 3.0), OUT / "ch06-fig05-pd-frames.png", camera="front", step_fn=_step, width=600, height=500))

if want_view():
    model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model)
    def _step2():
        data.ctrl[0] = pd_gravity(data); mujoco.mj_step(model, data)
    play(model, data, step_fn=_step2, duration=5.0, title="PD + gravity compensation (target 60 deg)")
