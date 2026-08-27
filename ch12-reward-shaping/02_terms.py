# labs/ch12-reward-shaping/02_terms.py
"""12.2 Speed tracking, torso stability, foot contact pattern.

The chain R1 (heading) → R2 (track a 0.5 m/s target speed instead of maximizing forward speed) → R3 (+ contact match). Speed over time and gait diagrams show the effect of each term.
Figures: out/ch12-fig03-speed-tracking.png, out/ch12-fig04-contact-diagrams.png
Run: uv run python 02_terms.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadbook.gait import LEG_ORDER, contacts_to_intervals, plot_gait_diagram
from train_lib import OUT, load_policy, run_episode

names = ("R1_heading", "R2_track", "R3_contact")
M = {n: json.load(open(OUT / f"metrics_{n}.json")) for n in names}
print("5-seed mean                          speed[m/s]  yaw[°]  β(LF RF LH RH)             fragments/cycle(LF RF LH RH)     contact match  impact[N]  CoT")
for n, m in M.items():
    print(f"  {n:12s} {m['speed']:8.3f} {m['yaw_deg']:+7.0f}  {[round(b, 2) for b in m['duty']]}  {[round(f, 1) for f in m['fragments']]}   {m['contact_match']:.2f}     {m['impact']:5.0f}  {m['cot']:.2f}")

logs = {}
fig, ax = plt.subplots(figsize=(8, 3.4))
for n, c in zip(names, ("#2ca02c", "#1f77b4", "#9467bd")):
    model, venv, env = load_policy(n)
    lg = run_episode(venv, env, lambda o: model.predict(o, deterministic=True)[0], seed=7); logs[n] = lg; venv.close()
    k = 50; sm = np.convolve(lg["forward"], np.ones(k) / k, mode="valid")
    ax.plot(lg["t"][k - 1:], sm, color=c, lw=1.2, label=n)
ax.axhline(0.5, color="#d62728", ls=":", label="v_target 0.5 m/s"); ax.set_xlabel("time [s]"); ax.set_ylabel("forward speed [m/s] (0.5 s average)"); ax.set_title("speed over an episode: maximize vs track", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch12-fig03-speed-tracking.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, n in zip(axes, ("R2_track", "R3_contact")):
    t, c = logs[n]["t"], logs[n]["contact"]; T = 0.5
    t0 = [s for s, _ in contacts_to_intervals(t, c[:, 0]) if s > 3.0][0]; m = (t >= t0) & (t <= t0 + 2 * T)
    iv = {leg: [((s - t0) / T, (e - t0) / T) for s, e in contacts_to_intervals(t[m], c[m][:, i])] for i, leg in enumerate(LEG_ORDER)}
    plot_gait_diagram(ax, iv, x_max=2, color="#333333", title=f"{n} (measured)")
fig.tight_layout(); fig.savefig(OUT / "ch12-fig04-contact-diagrams.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch12-fig03-speed-tracking.png", OUT / "ch12-fig04-contact-diagrams.png")
