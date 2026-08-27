# labs/ch06-standing-control/04_stand_robust.py
"""6.4 A robot that keeps standing under disturbances.

Step 1) Load balance: the Chapter 5 standing pose was lopsided, 14 N on the front feet and 28 N on the hind feet. Vary the hip angle to balance the front and rear loads.
Step 2) Push limit: the largest side push the joint PD alone can survive.
Step 3) Widen the support polygon (feet apart) and lower the center of mass (crouch), and see how much the limit rises (the static stability margin of Section 3.5).
Step 4) Why "attitude feedback" (reading the torso tilt from the IMU and turning the joints) does not help, as long as the feet stay on the ground.
Figures: out/ch06-fig09-stance-balance.png, out/ch06-fig10-push-limit.png, out/ch06-fig11-push-frames.png
Run: uv run python ch06-standing-control/04_stand_robust.py
Add --view to push the feet-apart robot harder every second (--speed 0.5 recommended).
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
from quadbook.control import JointPD, critical_kd, joint_inertia_diag, torso_roll_pitch
from quadbook.render import play, want_view
from quadbook.robot import STAND_POSE, TOUCH_SENSORS, load

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
KP = 40.0


def make():
    model, data = load(model_path())
    pd = JointPD(KP, critical_kd(KP, joint_inertia_diag(model, data)))
    return model, data, pd


def pose(abd=0.0, hip=0.8, knee=-1.5):
    """Leg order LF, RF, LH, RH. Abduction must be + on the left and − on the right so that all four feet spread outward."""
    return np.array([abd, hip, knee, -abd, hip, knee, abd, hip, knee, -abd, hip, knee])


def settle(p, seconds=2.0):
    model, data, pd = make()
    data.qpos[7:19] = p; mujoco.mj_forward(model, data)
    while data.time < seconds:
        data.ctrl[:] = pd.torque(data, p); mujoco.mj_step(model, data)
    return model, data, pd


# ---- Step 1: load balance ----
print("Step 1  foot loads by hip angle (LF RF LH RH, N) and x of the CoM and the foot centroid")
hips = (0.8, 0.9, 1.0, 1.1)
ratios = []
for hip in hips:
    model, data, _ = settle(pose(hip=hip))
    f = np.array([data.sensor(n).data[0] for n in TOUCH_SENSORS])
    com_x = data.subtree_com[model.body("torso").id][0]
    feet_x = np.mean([data.geom(f"{l}_foot").xpos[0] for l in ("LF", "RF", "LH", "RH")])
    ratios.append(f[:2].sum() / f[2:].sum())
    print(f"        hip {hip:.1f}: {np.round(f, 1)}  front/rear {ratios[-1]:.2f}  CoM x {com_x:+.3f}  feet x {feet_x:+.3f}  height {data.body('torso').xpos[2]:.3f}")
HIP_BALANCED = hips[int(np.argmin(np.abs(np.array(ratios) - 1)))]
BALANCED = pose(hip=HIP_BALANCED)
print(f"        → balanced pose: hip {HIP_BALANCED} rad (quadbook.robot.STAND_POSE_BALANCED)")
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(hips, ratios, "o-"); ax.axhline(1.0, color="#999999", ls=":")
ax.set_xlabel("hip angle [rad]"); ax.set_ylabel("front / rear foot load"); ax.set_title("stance balance vs hip angle")
fig.tight_layout(); fig.savefig(OUT / "ch06-fig09-stance-balance.png", dpi=200); plt.close(fig)


# ---- Steps 2 and 3: push limit ----
def push_limit(p, target_fn=None, forces=range(10, 200, 10)):
    """Raise a +y push applied for 0.2 s in 10 N steps and return the largest force before the robot falls."""
    last = 0
    for F in forces:
        model, data, pd = settle(p, 1.0)
        torso = model.body("torso").id
        while data.time < 3.0:
            tgt = target_fn(data, p) if target_fn else p
            data.ctrl[:] = pd.torque(data, tgt)
            data.xfrc_applied[torso, :] = 0.0
            if 1.0 <= data.time < 1.2:
                data.xfrc_applied[torso, 1] = F
            mujoco.mj_step(model, data)
        if data.body("torso").xpos[2] < 0.2:
            return last
        last = F
    return last


cases = {
    "balanced stance": BALANCED,
    "feet apart 0.15 rad": pose(abd=0.15, hip=HIP_BALANCED),
    "feet apart 0.30 rad": pose(abd=0.30, hip=HIP_BALANCED),
    "crouched (hip 1.2, knee -2.1)": pose(hip=1.2, knee=-2.1),
}
print("Steps 2 and 3  push limit by stance (kp 40)")
limits = {}
for name, p in cases.items():
    model, data, _ = settle(p)
    width = abs(data.geom("LF_foot").xpos[1] - data.geom("RF_foot").xpos[1])
    limits[name] = push_limit(p)
    print(f"        {name:30s}: max survived {limits[name]:3d} N | lateral foot spacing {width:.2f} m, torso height {data.body('torso').xpos[2]:.3f} m")

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(list(limits.keys()), list(limits.values()), color="#4a90d9")
ax.set_xlabel("max side push survived, 0.2 s [N]"); ax.set_title("push limit vs stance (joint PD, kp 40)")
fig.tight_layout(); fig.savefig(OUT / "ch06-fig10-push-limit.png", dpi=200); plt.close(fig)


# ---- Step 4: why attitude feedback does not work ----
def attitude_feedback(k_roll):
    """Shift the abduction targets of all four legs in the same direction in proportion to the torso roll (an attempt to move the feet toward the tilt)."""
    def fn(data, p):
        roll, _ = torso_roll_pitch(data)
        q = p.copy(); q[0::3] += -k_roll * roll
        return q
    return fn

print("Step 4  IMU roll → abduction joint correction (balanced pose, kp 40)")
for k in (0.0, 0.5, 1.5):
    print(f"        k_roll {k:.1f}: max survived {push_limit(BALANCED, attitude_feedback(k) if k else None):3d} N")

# ---- Frames: same 60 N push, balanced pose vs feet apart 0.3 (0.5 s after the push) ----
frames = []
for name, p in (("balanced, 60 N", BALANCED), ("feet apart 0.3, 60 N", pose(abd=0.30, hip=HIP_BALANCED))):
    model, data, pd = settle(p, 1.0); torso = model.body("torso").id
    while data.time < 1.7:
        data.ctrl[:] = pd.torque(data, p); data.xfrc_applied[torso, :] = 0.0
        if 1.0 <= data.time < 1.2: data.xfrc_applied[torso, 1] = 60.0
        mujoco.mj_step(model, data)
    ren = mujoco.Renderer(model, height=500, width=700); ren.update_scene(data, camera="front")
    frames.append(R._label(ren.render().copy(), f"{name} (t = 1.7 s)")); ren.close()
Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch06-fig11-push-frames.png")
print("saved", OUT / "ch06-fig09-stance-balance.png", OUT / "ch06-fig10-push-limit.png", OUT / "ch06-fig11-push-frames.png")

if want_view():
    p = pose(abd=0.30, hip=HIP_BALANCED)
    model, data, pd = settle(p, 1.0); torso = model.body("torso").id
    def _step():
        data.ctrl[:] = pd.torque(data, p); data.xfrc_applied[torso, :] = 0.0
        k = int(data.time)
        if k >= 2 and data.time - k < 0.2:
            data.xfrc_applied[torso, 1] = 20.0 * (k - 1) * (1 if k % 2 else -1)   # 20, 40, 60, ... N, alternating left and right
        mujoco.mj_step(model, data)
    play(model, data, step_fn=_step, duration=9.0, title="Pushing the feet-apart robot harder and harder (20, 40, 60, ... N)")
