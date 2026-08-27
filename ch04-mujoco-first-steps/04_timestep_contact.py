# labs/ch04-mujoco-first-steps/04_timestep_contact.py
"""Physics engine essentials: verify timestep, integrator, and contact by experiment.

Experiment 1) Run a two-joint pendulum while increasing the timestep. Too large and the energy blows up.
Experiment 2) At timestep 0.04 s, switch the integrator (Euler / implicitfast / implicit / RK4).
Experiment 3) Drop a box and read the contents of the contact (mjContact). Change contact "softness" with solref.
Result figure: out/ch04-fig05-timestep-stability.png
Run: uv run python ch04-mujoco-first-steps/04_timestep_contact.py
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

PENDULUM = """
<mujoco model="double_pendulum">
  <option timestep="{dt}" integrator="{integrator}"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="0 -2.2 0.6" xyaxes="1 0 0 0 0 1"/>
    <!-- Stand: a post rising from the floor to the pivot plus a crossbar. Visual only, no collisions (contype 0) -->
    <geom type="cylinder" fromto="0 0.15 0 0 0.15 1" size="0.015" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0.15 1 0 0 1" size="0.012" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="sphere" pos="0 0 1" size="0.03" rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/>
    <body pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="1"/>
      <body pos="0 0 -0.4">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def run_pendulum(dt, integrator, duration=5.0):
    """Run an undamped double pendulum for duration seconds and return total energy over time."""
    model = mujoco.MjModel.from_xml_string(PENDULUM.format(dt=dt, integrator=integrator))
    data = mujoco.MjData(model)
    data.qpos[:] = [np.deg2rad(120), 0.0]
    n = int(duration / dt)
    t, e = [], []
    for i in range(n):
        mujoco.mj_step(model, data)
        if i % max(1, n // 500) == 0:
            # total energy = potential + kinetic (mjData.energy is only computed when the flag is enabled)
            mujoco.mj_energyPos(model, data)
            mujoco.mj_energyVel(model, data)
            t.append(data.time)
            e.append(data.energy[0] + data.energy[1])
        if not np.all(np.isfinite(data.qpos)) or np.abs(data.qvel).max() > 1e4:
            break   # diverged
    return np.array(t), np.array(e)


# ---- Experiment 1: timestep sweep (implicitfast fixed) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
for dt in [0.001, 0.005, 0.02, 0.05]:
    t, e = run_pendulum(dt, "implicitfast")
    ax.plot(t, e, label=f"timestep = {dt} s")
    print(f"[timestep {dt:5.3f}, implicitfast] energy after 5 s {e[-1]:8.2f} J (start {e[0]:.2f} J), samples {len(t)}")
ax.set_title("(a) integrator = implicitfast, varying timestep")
ax.set_xlabel("time [s]")
ax.set_ylabel("total energy [J]")
ax.set_ylim(-15, 40)
ax.legend(fontsize=8)

# ---- Experiment 2: integrator comparison (timestep fixed at 0.04: the edge where the Euler family breaks down) ----
ax = axes[1]
for integ in ["Euler", "implicitfast", "implicit", "RK4"]:
    t, e = run_pendulum(0.04, integ)
    ax.plot(t, e, label=integ)
    print(f"[timestep 0.040, {integ:12s}] energy after 5 s {e[-1]:10.2f} J, samples {len(t)}")
ax.set_title("(b) timestep = 0.04 s, varying integrator")
ax.set_xlabel("time [s]")
ax.set_ylim(-15, 80)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ch04-fig05-timestep-stability.png", dpi=200)
plt.close(fig)

# The double pendulum swinging (timestep 0.002, implicitfast)
_m = mujoco.MjModel.from_xml_string(PENDULUM.format(dt=0.002, integrator="implicitfast"))
_d = mujoco.MjData(_m); _d.qpos[:] = [np.deg2rad(120), 0.0]
print("saved", snapshots(_m, _d, (0.0, 0.4, 0.8, 1.2), OUT / "ch04-fig04-double-pendulum.png", camera="front", width=600, height=600))

# ---- Experiment 3: reading contacts and solref ----
BOX = """
<mujoco model="contact_demo">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 0.5">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1" solref="{solref}"/>
    </body>
  </worldbody>
</mujoco>
"""
print("\n=== Contact experiment: penetration depth vs solref ===")
for solref in ["0.02 1", "0.05 1", "0.2 1"]:
    model = mujoco.MjModel.from_xml_string(BOX.format(solref=solref))
    data = mujoco.MjData(model)
    for _ in range(1500):        # 3 s, long enough to come to rest
        mujoco.mj_step(model, data)
    c = data.contact[0]
    g1, g2 = model.geom(c.geom1).name, model.geom(c.geom2).name
    print(f"solref='{solref}': {data.ncon} contacts, first contact {g1}-{g2}, dist={c.dist*1000:6.2f} mm "
          f"(negative = penetration), normal={np.round(c.frame[:3], 2)}")

if want_view():
    _m = mujoco.MjModel.from_xml_string(PENDULUM.format(dt=0.002, integrator="implicitfast"))
    _d = mujoco.MjData(_m); _d.qpos[:] = [np.deg2rad(120), 0.0]
    play(_m, _d, duration=10.0, title="Double pendulum (timestep 0.002, implicitfast)")
