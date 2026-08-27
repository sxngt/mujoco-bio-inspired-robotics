# labs/ch06-standing-control/03_gain_tuning.py
"""6.3 Gain tuning: between stiff and wobbly.

Experiment A) Step response. Start the joints 0.3 rad away from the standing pose and let them return to it.
        Vary kp; compute kd from the joint inertia for a damping ratio ζ = 1 (critical damping).
        Metrics: settling time, overshoot, required torque (clipped when it exceeds the 12 N·m limit), rest height.
Experiment B) Same kp, with only ζ switched between 0.3 and 1.0, to see the role of the damper.
Experiment C) Control rate. Physics runs at 500 Hz (0.002 s); running the controller at 500/100/50 Hz makes a large kp chatter because of the delay.
Figures: out/ch06-fig06-gain-step.png, out/ch06-fig07-gain-frames.png, out/ch06-fig08-control-rate.png
Run: uv run python ch06-standing-control/03_gain_tuning.py
Add --view to watch the kp 40 response in the viewer (--speed 0.5 recommended).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import mujoco
import quadbook.render as R
from quadbook import model_path
from quadbook.control import JointPD, critical_kd, joint_inertia_diag
from quadbook.render import play, want_view
from quadbook.robot import STAND_POSE, joint_qpos, load

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

OFFSET = np.array([0.0, 0.3, -0.3] * 4)     # starting pose offset by hip +0.3, knee −0.3 rad


def step_response(kp, zeta=1.0, control_every=1, duration=2.0):
    """Response returning from the offset pose to STAND_POSE. The torque is recomputed only once every control_every steps."""
    model, data = load(model_path())
    kd = critical_kd(kp, joint_inertia_diag(model, data), zeta)
    pd = JointPD(kp, kd)
    data.qpos[7:19] = STAND_POSE + OFFSET
    mujoco.mj_forward(model, data)
    t, q_hip, z, tau_req = [], [], [], []
    torque = np.zeros(12)
    i = 0
    while data.time < duration:
        if i % control_every == 0:
            torque = pd.torque(data, STAND_POSE)
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        i += 1
        t.append(data.time); q_hip.append(joint_qpos(data)[1]); z.append(data.body("torso").xpos[2]); tau_req.append(np.abs(torque).max())
    t, q_hip, z, tau_req = map(np.array, (t, q_hip, z, tau_req))
    err = q_hip - STAND_POSE[1]
    settled = np.where(np.abs(err) > 0.02)[0]
    t_settle = t[settled[-1]] if len(settled) else 0.0
    overshoot = max(0.0, -err.min()) / 0.3 * 100
    jitter = np.degrees(q_hip[t > 1.0].std())          # std of the wobble after 1 s (chatter metric)
    return model, data, kd, t, q_hip, z, tau_req, t_settle, overshoot, jitter


# ---- Experiment A: kp sweep (ζ = 1, 500 Hz) ----
print("Exp A   kp     kd(hip)  settle    ovsht  torque req(max) rest height")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
frames = []
for kp in (10.0, 40.0, 160.0):
    model, data, kd, t, q, z, tau, ts, ov, _ = step_response(kp)
    print(f"        {kp:5.0f}  {kd[1]:6.2f}   {ts:5.2f} s  {ov:4.0f}%   {tau.max():6.1f} N·m     {z[-1]:.3f} m" + ("   ← clipped above the 12 limit" if tau.max() > 12 else ""))
    axes[0].plot(t, np.degrees(q), label=f"kp = {kp:.0f}, kd = {kd[1]:.1f}")
    axes[1].plot(t, z, label=f"kp = {kp:.0f}")
    ren = mujoco.Renderer(model, height=500, width=700); ren.update_scene(data, camera="side")
    frames.append(R._label(ren.render().copy(), f"kp = {kp:.0f} (t = 2 s)")); ren.close()
axes[0].axhline(np.degrees(STAND_POSE[1]), color="#999999", ls=":", lw=1)
axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("LF hip angle [deg]"); axes[0].set_title("(a) hip step response, critical damping"); axes[0].legend(fontsize=8)
axes[1].axhline(0.31, color="#999999", ls=":", lw=1)
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("torso height [m]"); axes[1].set_title("(b) torso height"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch06-fig06-gain-step.png", dpi=200); plt.close(fig)
Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch06-fig07-gain-frames.png")

# ---- Experiment B: same kp, different ζ ----
print("Exp B   kp 40, changing only the damping ratio")
for zeta in (0.3, 1.0):
    *_, ts, ov, _ = step_response(40.0, zeta=zeta)
    print(f"        ζ = {zeta}: settle {ts:.2f} s, overshoot {ov:.0f}%")

# ---- Experiment C: control rate × kp → chatter ----
print("Exp C   hip angle wobble after 1 s by control rate (std, °)")
rates = {1: "500 Hz", 5: "100 Hz", 10: "50 Hz"}
kps = (40.0, 400.0, 3000.0)
table = np.zeros((len(kps), len(rates)))
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, kp in enumerate(kps):
    for j, (every, name) in enumerate(rates.items()):
        model, data, kd, t, q, z, tau, ts, ov, jitter = step_response(kp, control_every=every)
        table[i, j] = jitter
        if kp == 400.0:
            ax.plot(t, np.degrees(q), lw=0.9, label=f"kp 400, control {name}")
    print(f"        kp {kp:5.0f}: " + "  ".join(f"{name} {table[i, j]:5.2f}°" for j, name in enumerate(rates.values())))
ax.axhline(np.degrees(STAND_POSE[1]), color="#999999", ls=":", lw=1)
ax.set_xlim(0.8, 2.0); ax.set_ylim(np.degrees(STAND_POSE[1]) - 8, np.degrees(STAND_POSE[1]) + 8)
ax.set_xlabel("time [s]"); ax.set_ylabel("LF hip angle [deg]"); ax.set_title("(c) same gains, slower control loop: chatter"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch06-fig08-control-rate.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch06-fig06-gain-step.png", OUT / "ch06-fig07-gain-frames.png", OUT / "ch06-fig08-control-rate.png")

if want_view():
    model, data = load(model_path())
    pd = JointPD(40.0, critical_kd(40.0, joint_inertia_diag(model, data)))
    data.qpos[7:19] = STAND_POSE + OFFSET; mujoco.mj_forward(model, data)
    def _step():
        data.ctrl[:] = pd.torque(data, STAND_POSE); mujoco.mj_step(model, data)
    play(model, data, step_fn=_step, duration=4.0, title="kp 40, critical damping: from the offset pose to the standing pose")
