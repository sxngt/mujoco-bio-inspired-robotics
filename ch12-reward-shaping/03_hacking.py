# labs/ch12-reward-shaping/03_hacking.py
"""12.3 A museum of strange gaits: a casebook of reward hacking failures.

Four deliberately misdesigned rewards (2M steps each); we look at what each trained policy learned.
H1 forward only (no attitude or energy penalty; it runs in circles) / H2 alive bonus + energy penalty (no forward term) / H3 excessive touchdown impact penalty / H4 excessive contact match reward
Figures: out/ch12-fig05-hacking-frames.png, out/ch12-fig06-hacking-bars.png
Run: uv run python 03_hacking.py   (--view --speed 0.5 H1: replay a specific case, e.g. --view H2)
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from quadbook.render import play, want_view
from train_lib import CONFIGS, OUT, load_policy, run_episode

names = ("H1_forward_only", "H2_alive_energy", "H3_impact_heavy", "H4_contact_heavy")
M = {n: json.load(open(OUT / f"metrics_{n}.json")) for n in names}
M["ppo_first"] = json.load(open(OUT / "metrics_ppo_first.json"))
print("5-seed mean                     speed[m/s]  falls  yaw[°]  fragments/cycle           impact[N]  |δ|[°]  contact match  CoT")
for n in ("ppo_first",) + names:
    m = M[n]; print(f"  {n:16s} {m['speed']:8.3f}  {m['fall_rate']:5.0%}  {m['yaw_deg']:+6.0f}  {[round(f, 1) for f in m['fragments']] if m['fragments'] else 'n/a'}   {m['impact']:5.0f}   {m['delta_deg']:4.1f}    {m['contact_match']:.2f}   {m['cot']:.2f}")

# Dissecting H1's turning: does it turn once or keep circling? Is the left/right correction symmetric?
model, venv, env = load_policy("H1_forward_only"); pol = lambda o: model.predict(o, deterministic=True)[0]
print("H1 dissection (per seed): yaw rate, body-frame forward speed, circle radius v/ω, left/right difference of the knee correction")
for seed in (7, 8, 9):
    lg = run_episode(venv, env, pol, seed=seed); t = lg["t"]; yaw = np.degrees(np.unwrap(lg["yaw"])); rate = np.gradient(yaw, t)[t > 2].mean()
    v = lg["forward"].mean(); d = np.degrees(lg["delta"]).mean(axis=0)
    print(f"  seed {seed}: yaw rate {rate:+.1f}°/s ({yaw[-1]:+.0f}° after 10 s) | body-frame forward {v:.3f} m/s | radius {v/abs(np.radians(rate)):.1f} m | mean knee δ LF {d[2]:+.1f}° RF {d[5]:+.1f}° LH {d[8]:+.1f}° RH {d[11]:+.1f}°")
venv.close()

frames = []
for n in names:
    model, venv, env = load_policy(n)
    lg = run_episode(venv, env, lambda o: model.predict(o, deterministic=True)[0], seed=7, frames_at=(3.0,), label=n, max_t=4.0)
    if not lg["frames"]:   # fell before 3 s: use the last instant
        lg = run_episode(venv, env, lambda o: model.predict(o, deterministic=True)[0], seed=7, frames_at=(round(lg["t"][-1] - 0.05, 2),), label=n + " (before fall)", max_t=4.0)
    frames.append(lg["frames"][0] if lg["frames"] else env.render()); venv.close()
Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch12-fig05-hacking-frames.png")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
allnames = ("ppo_first",) + names; cols = ("#ff7f0e", "#d62728", "#d62728", "#d62728", "#d62728")
axes[0].bar(allnames, [M[n]["speed"] for n in allnames], color=cols); axes[0].set_title("speed [m/s]", fontsize=10)
axes[1].bar(allnames, [abs(M[n]["yaw_deg"]) for n in allnames], color=cols); axes[1].set_title("|yaw| after 10 s [deg]", fontsize=10)
axes[2].bar(allnames, [M[n]["delta_deg"] for n in allnames], color=cols); axes[2].set_title("mean |delta| [deg]", fontsize=10)
for ax in axes: ax.tick_params(axis="x", labelsize=7, rotation=15)
fig.tight_layout(); fig.savefig(OUT / "ch12-fig06-hacking-bars.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch12-fig05-hacking-frames.png", OUT / "ch12-fig06-hacking-bars.png")

if want_view():
    pick = next((a for a in sys.argv[1:] if a.startswith("H")), "H1")
    n = next(x for x in names if x.startswith(pick)); model, venv, env = load_policy(n)
    state = {"obs": venv.reset()}
    def _step():
        a = model.predict(state["obs"], deterministic=True)[0]; state["obs"], _, _, _ = venv.step(a)
    play(env.model, env.data, step_fn=_step, duration=10.0, title=f"{n}: {CONFIGS[n]['weights']}")
