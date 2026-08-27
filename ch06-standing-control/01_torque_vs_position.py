# labs/ch06-standing-control/01_torque_vs_position.py
"""6.1 Torque control vs position control.

Experiment A) Apply a constant torque to the hip joint of a test-rig leg. Without feedback the joint does not "know" the target angle:
        it simply stops at the angle where the gravity torque balances the motor torque. Cut the torque and it falls.
Experiment B) Drive the same joint to 0.8 rad with a position actuator (built-in PD). With feedback it reaches the target.
Experiment C) Drop the full robot from 0.4 m. Compare the landing impact of a stiff PD (kp 400) and a compliant PD (kp 40).
Figures: out/ch06-fig01-torque-vs-position.png, out/ch06-fig02-landing-frames.png, out/ch06-fig03-landing-stiffness.png
Run: uv run python ch06-standing-control/01_torque_vs_position.py
Add --view to watch the compliant landing of experiment C in the viewer (macOS: uv run mjpython ... --view --speed 0.5).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.control import JointPD, critical_kd, joint_inertia_diag
from quadbook.render import play, snapshots, want_view
from quadbook.robot import STAND_POSE, TOUCH_SENSORS, load

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

LEG_XML = """
<mujoco model="leg_rig">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <default><joint damping="0.2" armature="0.01"/><default class="leg"><geom type="capsule" rgba="0.55 0.55 0.6 1"/></default></default>
  <worldbody>
    <light pos="1 1 3" dir="-0.3 -0.3 -1"/>
    <camera name="side" pos="0.75 -0.75 0.3" xyaxes="0.7 0.7 0 0 0 1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="hip_mount" pos="0 0 0.5">
      <geom type="box" size="0.03 0.03 0.03" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <body name="thigh" pos="0 0.04 0">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.2 2.4"/>
        <geom class="leg" fromto="0 0 0 0 0 -0.2" size="0.025" mass="0.6"/>
        <body name="calf" pos="0 0 -0.2">
          <joint name="knee" type="hinge" axis="0 1 0" range="-2.6 -0.3"/>
          <geom class="leg" fromto="0 0 0 0 0 -0.2" size="0.018" mass="0.3"/>
          <geom name="foot" type="sphere" pos="0 0 -0.2" size="0.02" mass="0.05" rgba="0.15 0.15 0.15 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    {hip_actuator}
    <position name="knee_pos" joint="knee" kp="30" kv="1"/>
  </actuator>
</mujoco>
"""


def run_leg(hip_actuator, hip_cmd, duration=3.0):
    """Feed the value returned by hip_cmd(t) to the hip actuator and hold the knee at -1.5 rad."""
    model = mujoco.MjModel.from_xml_string(LEG_XML.format(hip_actuator=hip_actuator))
    data = mujoco.MjData(model)
    t, q = [], []
    while data.time < duration:
        data.ctrl[0] = hip_cmd(data.time)
        data.ctrl[1] = -1.5
        mujoco.mj_step(model, data)
        t.append(data.time)
        q.append(data.qpos[0])
    return np.array(t), np.array(q)


# ---- Experiment A: torque only (motor) ----
TORQUE = 1.0
tA, qA = run_leg('<motor name="hip_motor" joint="hip" ctrlrange="-12 12"/>', lambda t: TORQUE if t < 2.0 else 0.0)
# Predict the balance angle: ask MuJoCo (qfrc_bias) for the gravity torque on the hip at angle theta and find the theta where it equals the motor torque
m_rig = mujoco.MjModel.from_xml_string(LEG_XML.format(hip_actuator='<motor name="hip_motor" joint="hip" ctrlrange="-12 12"/>'))
d_rig = mujoco.MjData(m_rig)
thetas = np.linspace(0, np.pi / 2, 900)
grav = []
for th in thetas:
    d_rig.qpos[:] = [th, -1.5]; d_rig.qvel[:] = 0
    mujoco.mj_forward(m_rig, d_rig)
    grav.append(d_rig.qfrc_bias[0])            # bias force at rest = gravity torque the motor must overcome
grav = np.array(grav)
theta_pred = thetas[np.argmin(np.abs(grav - TORQUE))]   # angle where the bias force equals the motor torque
theta_meas = qA[(tA > 1.5) & (tA < 2.0)].mean()
print(f"[A] torque {TORQUE} N·m: predicted balance angle {np.degrees(theta_pred):.1f}°, measured {np.degrees(theta_meas):.1f}° | angle at 3 s, after the torque is cut {np.degrees(qA[-1]):.1f}°")

# ---- Experiment B: position actuator ----
tB, qB = run_leg('<position name="hip_pos" joint="hip" kp="20" kv="1" ctrlrange="-3 3"/>', lambda t: 0.8 if t < 2.0 else 0.0)
print(f"[B] position(kp 20): angle at 2 s {np.degrees(qB[(tB > 1.5) & (tB < 2.0)].mean()):.1f}° (target 45.8°), at 3 s after the target is switched to 0 {np.degrees(qB[-1]):.1f}°")

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(tA, np.degrees(qA), label="(A) motor: constant 1 N.m for 2 s, then 0")
ax.plot(tB, np.degrees(qB), label="(B) position: target 0.8 rad for 2 s, then 0")
ax.axhline(np.degrees(theta_pred), color="#999999", ls=":", lw=1, label="gravity balance angle (predicted)")
ax.axhline(np.degrees(0.8), color="#bbbbbb", ls="--", lw=1)
ax.set_xlabel("time [s]"); ax.set_ylabel("hip angle [deg]"); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
ax.set_title("torque without feedback vs position with built-in feedback")
fig.tight_layout(); fig.savefig(OUT / "ch06-fig01-torque-vs-position.png", dpi=200); plt.close(fig)

# ---- Experiment C: landing impact, stiff PD vs compliant PD ----
def drop(kp, height=0.4, duration=1.5):
    model, data = load(model_path())
    kd = critical_kd(kp, joint_inertia_diag(model, data), zeta=1.0)
    pd = JointPD(kp, kd)
    data.qpos[2] = height
    mujoco.mj_forward(model, data)
    t, z, touch, tau = [], [], [], []
    while data.time < duration:
        data.ctrl[:] = pd.torque(data, STAND_POSE)
        mujoco.mj_step(model, data)
        t.append(data.time); z.append(data.body("torso").xpos[2])
        touch.append(sum(data.sensor(n).data[0] for n in TOUCH_SENSORS))
        tau.append(np.abs(data.ctrl).max())
    return model, data, kd, np.array(t), np.array(z), np.array(touch), np.array(tau)


results = {}
for kp in (40.0, 400.0):
    m, d, kd, t, z, touch, tau = drop(kp)
    results[kp] = (t, z, touch, tau)
    print(f"[C] kp={kp:>5.0f}, kd≈{kd[1]:.2f}: peak total contact force {touch.max():6.0f} N (body weight 84 N), peak torque {tau.max():5.1f} N·m (limit 12), rest height {z[-1]:.3f} m")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for kp, (t, z, touch, tau) in results.items():
    axes[0].plot(t, touch, label=f"kp = {kp:.0f}")
    axes[1].plot(t, z, label=f"kp = {kp:.0f}")
axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("total foot force [N]"); axes[0].set_title("(a) landing impact"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("torso height [m]"); axes[1].set_title("(b) torso height"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch06-fig03-landing-stiffness.png", dpi=200); plt.close(fig)

# Landing frames (compliant landing)
m, d = load(model_path()); d.qpos[2] = 0.4
kd = critical_kd(40.0, joint_inertia_diag(m, d)); pd = JointPD(40.0, kd)
def _step():
    d.ctrl[:] = pd.torque(d, STAND_POSE); mujoco.mj_step(m, d)
print("saved", snapshots(m, d, (0.0, 0.15, 0.25, 0.6), OUT / "ch06-fig02-landing-frames.png", camera="side", step_fn=_step, width=700, height=500))
print("saved", OUT / "ch06-fig01-torque-vs-position.png", OUT / "ch06-fig03-landing-stiffness.png")

if want_view():
    m, d = load(model_path()); d.qpos[2] = 0.4
    pd = JointPD(40.0, critical_kd(40.0, joint_inertia_diag(m, d)))
    def _step2():
        d.ctrl[:] = pd.torque(d, STAND_POSE); mujoco.mj_step(m, d)
    play(m, d, step_fn=_step2, duration=5.0, title="Landing with a compliant PD (kp 40)")
