# labs/ch07-first-gait/02_cpg_oscillator.py
"""7.2 This is a CPG: coupled phase oscillators.

(a) Start the four leg phases at random. With coupling k = 0 the phase differences stay as they are; with k > 0 they are pulled to the trot phase table.
(b) At 3 s, knock the LH leg phase off by 0.25 cycle. With coupling it comes back.
(c) One cycle of commands from the oscillator → foot trajectory → IK pipeline: foot height (the commanded gait diagram) and LF joint targets.
Figures: out/ch07-fig03-phase-coupling.png, out/ch07-fig04-cpg-pipeline.png
Run: uv run python ch07-first-gait/02_cpg_oscillator.py   (--view: a robot held in the air paddles its legs in time with the oscillators)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator, PhaseOscillators
from quadbook.gait import GAIT_PHASES, LEG_ORDER
from quadbook.render import play, want_view
from quadbook.robot import load

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
DT, F = 0.01, 2.0
rng = np.random.default_rng(7)
init = rng.random(4)
print("Initial phases (LF RF LH RH):", np.round(init, 2), "| trot target phases:", [GAIT_PHASES["trot"][l] for l in LEG_ORDER])


def run(k, perturb=True, T=6.0):
    osc = PhaseOscillators(F, GAIT_PHASES["trot"], coupling=k, init=init)
    t, rel = [], []
    for i in range(int(T / DT)):
        if perturb and abs(i * DT - 3.0) < DT / 2:
            osc.theta[2] = (osc.theta[2] + 0.25) % 1.0        # knock the LH leg off by 1/4 cycle
        th = osc.step(DT)
        t.append(i * DT); rel.append(((th - th[0]) + 0.25) % 1.0 - 0.25)      # phase relative to LF, shown in [-0.25, 0.75)
    return np.array(t), np.array(rel)


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, k in zip(axes, (0.0, 2.0)):
    t, rel = run(k)
    for i, leg in enumerate(LEG_ORDER[1:], start=1):
        ax.plot(t, rel[:, i], label=f"{leg} - LF")
    for v in (0.0, 0.5):
        ax.axhline(v, color="#999999", ls=":", lw=1)
    ax.axvline(3.0, color="#d62728", ls="--", lw=1)
    ax.set_title(f"coupling k = {k:g}" + ("  (no coupling)" if k == 0 else "  (pulled to trot pattern)"), fontsize=10)
    ax.set_xlabel("time [s]"); ax.set_ylim(-0.27, 0.77)
    err = np.abs(((rel[-1] - np.array([GAIT_PHASES["trot"][l] for l in LEG_ORDER])) + 0.5) % 1.0 - 0.5)
    conv = t[np.where(np.abs(((rel - np.array([GAIT_PHASES['trot'][l] for l in LEG_ORDER])) + 0.5) % 1.0 - 0.5).max(axis=1) < 0.02)[0]]
    first = conv[conv < 3.0]; after = conv[conv > 3.0]
    print(f"k = {k:g}: max error from the target phase differences after 6 s {err.max():.3f} cycle | converged (error < 0.02) " + (f"at {first[0]:.2f} s, reconverged after the 3 s disturbance at {after[0]:.2f} s" if len(first) and len(after) else "never"))
axes[0].set_ylabel("phase relative to LF [cycle]"); axes[1].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "ch07-fig03-phase-coupling.png", dpi=200); plt.close(fig)

# ---- (c) one cycle produced by the pipeline ----
gen = GaitGenerator("trot", freq=F, duty=0.5, step_length=0.08, step_height=0.04)
ts = np.arange(0, 1.0 / F, DT / 2)
foot_h, q = [], []
for _ in ts:
    q.append(gen.targets(DT / 2)); foot_h.append(gen.foot_height_cmd(gen.osc.theta))
foot_h, q = np.array(foot_h), np.array(q)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for i, leg in enumerate(LEG_ORDER):
    axes[0].plot(ts * F, 1000 * foot_h[:, i] + 45 * (3 - i), label=leg)
axes[0].set_yticks([45 * (3 - i) for i in range(4)]); axes[0].set_yticklabels(LEG_ORDER)
axes[0].set_xlabel("gait cycle"); axes[0].set_title("(a) commanded foot height per leg (flat = stance)", fontsize=10)
axes[1].plot(ts * F, q[:, 1], label="LF hip target"); axes[1].plot(ts * F, q[:, 2], label="LF knee target")
axes[1].set_xlabel("gait cycle"); axes[1].set_ylabel("joint target [rad]"); axes[1].set_title("(b) joint targets from IK", fontsize=10); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch07-fig04-cpg-pipeline.png", dpi=200); plt.close(fig)
stance = (foot_h < 1e-6).mean(axis=0)
print("Commanded duty factor (per leg):", np.round(stance, 2), "| joint target range hip", np.round([q[:,1].min(), q[:,1].max()], 2), "knee", np.round([q[:,2].min(), q[:,2].max()], 2))
print("saved", OUT / "ch07-fig03-phase-coupling.png", OUT / "ch07-fig04-cpg-pipeline.png")

if want_view():
    model, data = load(model_path()); model.opt.gravity[:] = 0; data.qpos[2] = 0.45
    from quadbook.sim import standard_pd
    pd = standard_pd(model, data)
    gen = GaitGenerator("trot", freq=1.0, init_phase=init)
    def _step():
        data.ctrl[:] = pd.torque(data, gen.targets(model.opt.timestep)); mujoco.mj_step(model, data)
    play(model, data, step_fn=_step, duration=10.0, title="In the air: randomly started phases lock into trot (1 Hz)")
