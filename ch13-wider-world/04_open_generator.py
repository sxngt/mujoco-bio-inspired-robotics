# labs/ch13-wider-world/04_open_generator.py
"""13.1 (second half) Opening the generator to the policy: at a 0.8 m/s target, residual only (12-dim) vs also modulating frequency and step length (14-dim).

An experiment on the limit stated at the end of Chapter 12 ('as far as it goes while the policy cannot change the generator constants'). Traces of the frequency and step length chosen by the policy, and the dynamic similarity coordinates of Section 8.4.
Figures: out/ch13-fig03-modulation.png, out/ch13-fig04-froude-modulated.png
Run: uv run python 04_open_generator.py   (after 00_train_all.py)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadbook.analysis import alexander_stride, froude
from train_lib import OUT, load_policy, run_episode

M = {n: json.load(open(OUT / f"metrics_{n}.json")) for n in ("B_residual_08", "B_modulate_08")}
print("5-seed mean (target 0.8 m/s)    speed   yaw[°]  falls   β(LF RF LH RH)             fragments        impact[N] CoT   freq[Hz]    step L[m]  measured stride λ[m]")
for n, m in M.items():
    print(f"  {n:14s} {m['speed']:.3f}  {m['yaw_deg']:+6.0f}  {m['fall_rate']:5.0%}  {[round(b, 2) for b in m['duty']]}  {[round(f, 1) for f in m['fragments']]}   {m['impact']:5.0f}  {m['cot']:.2f}   {m['freq']:.2f}       {m['step_length']:.3f}      {m['stride']:.3f}")

logs = {}
for n in ("B_residual_08", "B_modulate_08"):
    model, venv, env = load_policy(n); logs[n] = run_episode(venv, env, lambda o: model.predict(o, deterministic=True)[0], seed=7); venv.close()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for n, c in (("B_residual_08", "#8c564b"), ("B_modulate_08", "#17becf")):
    lg = logs[n]; k = 50; sm = np.convolve(lg["forward"], np.ones(k) / k, mode="valid")
    axes[0].plot(lg["t"][k - 1:], sm, color=c, label=n); axes[1].plot(lg["t"], lg["freq"], color=c, label=n); axes[2].plot(lg["t"], lg["step_length"], color=c, label=n)
axes[0].axhline(0.8, color="#d62728", ls=":", label="v_target 0.8"); axes[0].set_title("(a) forward speed", fontsize=10); axes[0].set_xlabel("time [s]"); axes[0].legend(fontsize=7)
axes[1].set_title("(b) generator frequency chosen by the policy [Hz]", fontsize=10); axes[1].set_xlabel("time [s]"); axes[1].set_ylim(1.4, 3.1)
axes[2].set_title("(c) generator step length chosen [m]", fontsize=10); axes[2].set_xlabel("time [s]"); axes[2].set_ylim(0.04, 0.15)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig03-modulation.png", dpi=200); plt.close(fig)

L = 0.30
fig, ax = plt.subplots(figsize=(7, 4)); frs = np.logspace(-2, 0.7, 100)
ax.plot(frs, alexander_stride(frs), color="#999999", label="Alexander & Jayes: 2.4 Fr^0.34"); ax.axvline(0.5, color="#bbbbbb", ls=":"); ax.text(0.5, 3.9, "walk / trot", fontsize=8, color="#666666", ha="center")
pts = [("baseline trot (ch7)", 0.286, 0.145, "#999999"), ("final policy (ch12)", 0.486, 0.238, "#17becf"), ("residual only, 0.8 m/s", M["B_residual_08"]["speed"], M["B_residual_08"]["stride"], "#8c564b"), ("generator opened, 0.8 m/s", M["B_modulate_08"]["speed"], M["B_modulate_08"]["stride"], "#d62728")]
for name, v, stride, c in pts:
    fr = froude(v, L); ax.plot(fr, stride / L, "o", color=c, ms=8); ax.annotate(name, (fr, stride / L), textcoords="offset points", xytext=(6, -3), fontsize=8)
    print(f"  {name:26s}: Fr {fr:.3f}, relative stride {stride/L:.2f} (Alexander prediction {alexander_stride(fr):.2f})")
for name, (Lh, v) in {"dog walk": (0.5, 1.0), "dog trot": (0.5, 2.5)}.items():
    fr = froude(v, Lh); ax.plot(fr, alexander_stride(fr), "s", color="#d62728"); ax.annotate(name, (fr, alexander_stride(fr)), textcoords="offset points", xytext=(6, -3), fontsize=8)
ax.set_xscale("log"); ax.set_ylim(0, 4.3); ax.set_xlabel("Froude number"); ax.set_ylabel("relative stride length"); ax.set_title("dynamic similarity with the generator opened", fontsize=10); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ch13-fig04-froude-modulated.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch13-fig03-modulation.png", OUT / "ch13-fig04-froude-modulated.png")
