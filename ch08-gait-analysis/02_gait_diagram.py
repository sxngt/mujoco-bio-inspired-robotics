# labs/ch08-gait-analysis/02_gait_diagram.py
"""8.2 Drawing my robot's gait diagram.

Read the contact log saved by 01, draw the measured gait diagram with the Chapter 3 tool (quadbook.gait.plot_gait_diagram), and place it next to the diagram the generator commanded.
The horizontal axis is in 'cycles', normalized by the commanded period T = 1/f. Two cycles are drawn.
Figures: out/ch08-fig03-trot-diagram.png, out/ch08-fig04-walk-diagram.png
Run: uv run python ch08-gait-analysis/02_gait_diagram.py   (--view --speed 0.5: replays walk 1 Hz)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator
from quadbook.gait import LEG_ORDER, contacts_to_intervals, plot_gait_diagram
from quadbook.render import play, want_view
from quadbook.robot import load
from quadbook.sim import WALK_KP, standard_pd

OUT = Path(__file__).resolve().parent / "out"
L = np.load(OUT / "logs.npz")


def measured_intervals(name, t0, n_cycles=2):
    """Measured contact intervals over n_cycles cycles from t0, in cycle units."""
    t, c, f = L[f"{name}_t"], L[f"{name}_contact"], float(L[f"{name}_freq"])
    T = 1.0 / f
    m = (t >= t0) & (t <= t0 + n_cycles * T)
    return {leg: [((s - t0) / T, (e - t0) / T) for s, e in contacts_to_intervals(t[m], c[m][:, i])] for i, leg in enumerate(LEG_ORDER)}


def commanded_intervals(name, t0, n_cycles=2):
    t, fc, f = L[f"{name}_t"], L[f"{name}_foot_cmd"], float(L[f"{name}_freq"])
    T = 1.0 / f
    m = (t >= t0) & (t <= t0 + n_cycles * T)
    return {leg: [((s - t0) / T, (e - t0) / T) for s, e in contacts_to_intervals(t[m], fc[m][:, i] < 1e-6)] for i, leg in enumerate(LEG_ORDER)}


def lf_touchdown_after(name, t_from):
    """Time of the first LF touchdown after t_from (the diagram's zero point)."""
    t, c = L[f"{name}_t"], L[f"{name}_contact"][:, 0]
    starts = [s for s, _ in contacts_to_intervals(t, c) if s > t_from]
    return starts[0]


for name, title, fig_name in (("trot2", "trot 2 Hz", "ch08-fig03-trot-diagram.png"), ("walk1", "walk 1 Hz", "ch08-fig04-walk-diagram.png")):
    t0 = lf_touchdown_after(name, 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    plot_gait_diagram(axes[0], commanded_intervals(name, t0), x_max=2, color="#999999", title=f"(a) {title}: commanded")
    plot_gait_diagram(axes[1], measured_intervals(name, t0), x_max=2, color="#333333", title=f"(b) {title}: measured (contact list)")
    fig.tight_layout(); fig.savefig(OUT / fig_name, dpi=200); plt.close(fig)
    meas = measured_intervals(name, t0); cmd = commanded_intervals(name, t0)
    print(f"{title}: diagram zero = LF touchdown at {t0:.3f} s")
    for leg in LEG_ORDER:
        print(f"    {leg}: commanded stance intervals " + ", ".join(f"[{s:.2f}, {e:.2f}]" for s, e in cmd[leg]) + "  | measured " + ", ".join(f"[{s:.2f}, {e:.2f}]" for s, e in meas[leg]))

# trot 1 Hz: the foot lifting off at the end of stance
t0 = lf_touchdown_after("trot1", 3.0)
meas = measured_intervals("trot1", t0)
print("trot 1 Hz: LF measured stance intervals " + ", ".join(f"[{s:.2f}, {e:.2f}]" for s, e in meas["LF"]) + "  (commanded: [0.00, 0.50], [1.00, 1.50])")
print("saved", OUT / "ch08-fig03-trot-diagram.png", OUT / "ch08-fig04-walk-diagram.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="walk", freq=1.0, duty=0.75, step_length=0.08, step_height=0.04)
    state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    def _step():
        if data.time >= 0.5 and state["i"] % 5 == 0: state["q"], state["qd"] = gen.targets_with_velocity(model.opt.timestep * 5)
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
    play(model, data, step_fn=_step, duration=8.0, title="walk 1 Hz: one foot at a time in the order LF → RH → RF → LH")
