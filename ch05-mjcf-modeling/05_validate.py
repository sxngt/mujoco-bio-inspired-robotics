# labs/ch05-mjcf-modeling/05_validate.py
"""Model validation: drop it, shake it, and stay suspicious.

Test 1) Drop with no control: the legs should fold and the robot should collapse (meaning the joints are free).
Test 2) Drop while PD holds the pose: it should stand.
Test 3) Push the standing robot's torso from the side (xfrc_applied): it should wobble and recover, or fall if the force is large.
Test 4) Suspicion checklist: symmetry, joints within range, penetration depth, energy blow-up, contact count.
Output figure: out/ch05-fig08-validation.png
Run: uv run python ch05-mjcf-modeling/05_validate.py
Add --view to watch the same scene live in the viewer window (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # Draw to files only, no display (also works in the mjpython worker thread)
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.render import play, snapshots, want_view
from quadbook.robot import STAND_POSE, foot_contacts, joint_qpos, load, pd_torque

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def run(controlled: bool, push_N: float = 0.0, push_at: float = 1.0, push_dur: float = 0.2, duration: float = 3.0):
    model, data = load(model_path())
    data.qpos[2] = 0.4
    mujoco.mj_forward(model, data)
    t, z, roll = [], [], []
    while data.time < duration:
        data.ctrl[:] = pd_torque(data, STAND_POSE) if controlled else 0.0
        data.xfrc_applied[model.body("torso").id, :] = 0.0
        if push_at <= data.time < push_at + push_dur:
            data.xfrc_applied[model.body("torso").id, 1] = push_N     # Push in the +y direction (left)
        mujoco.mj_step(model, data)
        t.append(data.time)
        z.append(data.body("torso").xpos[2])
        # Torso roll angle: the angle between the torso y axis and the world z axis
        ymat = data.body("torso").xmat.reshape(3, 3)[:, 1]
        roll.append(np.degrees(np.arcsin(np.clip(ymat[2], -1, 1))))
    return model, data, np.array(t), np.array(z), np.array(roll)


# Tests 1 and 2
m0, d0, t0, z0, _ = run(controlled=False)
m1, d1, t1, z1, _ = run(controlled=True)
print(f"[Test 1] no control: torso height after 3 s {z0[-1]:.3f} m (below the standing height 0.31 m means it collapsed)")
print(f"[Test 2] PD holds  : torso height after 3 s {z1[-1]:.3f} m, foot contacts {foot_contacts(m1, d1)}")

# Test 3: two push strengths
results = {}
for push in (20.0, 60.0):
    m, d, t, z, roll = run(controlled=True, push_N=push)
    results[push] = (t, z, roll)
    fell = z[-1] < 0.2
    print(f"[Test 3] push torso with {push:.0f} N for 0.2 s: max roll {np.abs(roll).max():.1f}°, height after 3 s {z[-1]:.3f} m → {'fell' if fell else 'recovered'}")

# Test 4: suspicion checklist
m, d = load(model_path())
for _ in range(1000):
    d.ctrl[:] = pd_torque(d, STAND_POSE)
    mujoco.mj_step(m, d)
q = joint_qpos(d)
lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]          # index 0 is the freejoint
inside = np.all((q >= lo) & (q <= hi))
pen = min((d.contact[i].dist for i in range(d.ncon)), default=0.0)
mujoco.mj_energyPos(m, d); mujoco.mj_energyVel(m, d)
print(f"[Test 4] joints within range: {inside} | max penetration {pen*1000:.2f} mm | contacts {d.ncon} | kinetic energy {d.energy[1]:.4f} J (nearly 0 at rest)")
print(f"        max joint torque while standing {np.abs(d.ctrl).max():.2f} N·m (limit 12)")

# Picture the three outcomes: collapsed without control / standing with PD / fell at 60 N (each after 3 s)
from PIL import Image
import quadbook.render as _r
_frames = []
for ctrl, push in ((False, 0.0), (True, 0.0), (True, 60.0)):
    _m, _d, *_ = run(controlled=ctrl, push_N=push)
    _ren = mujoco.Renderer(_m, height=500, width=700); _ren.update_scene(_d, camera="side")
    _frames.append(_r._label(_ren.render().copy(), ["no control", "PD holds pose", "pushed 60 N"][len(_frames)])); _ren.close()
Image.fromarray(np.concatenate(_frames, axis=1)).save(OUT / "ch05-fig07-validation-frames.png"); print("saved", OUT / "ch05-fig07-validation-frames.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(t0, z0, label="no control (joints free)")
axes[0].plot(t1, z1, label="PD holds standing pose")
axes[0].axhline(0.31, color="#999999", ls=":", lw=1)
axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("torso height [m]"); axes[0].set_title("(a) drop test"); axes[0].legend(fontsize=8)
for push, (t, z, roll) in results.items():
    axes[1].plot(t, roll, label=f"push {push:.0f} N for 0.2 s")
axes[1].axvspan(1.0, 1.2, color="#eeeeee")
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("torso roll [deg]"); axes[1].set_title("(b) push test"); axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ch05-fig08-validation.png", dpi=200)
print("saved", OUT / "ch05-fig08-validation.png")

if want_view():
    # Watch it all at once: drop → 20 N push at 1 s (recovers) → 60 N push at 3 s (falls over)
    m2, d2 = load(model_path()); d2.qpos[2] = 0.4
    torso = m2.body("torso").id
    def _step():
        d2.ctrl[:] = pd_torque(d2, STAND_POSE)
        d2.xfrc_applied[torso, :] = 0.0
        if 1.0 <= d2.time < 1.2: d2.xfrc_applied[torso, 1] = 20.0
        if 3.0 <= d2.time < 3.2: d2.xfrc_applied[torso, 1] = 60.0
        mujoco.mj_step(m2, d2)
    play(m2, d2, step_fn=_step, duration=6.0, title="Validation: drop → 20 N push (recovers) → 60 N push (falls over)")
