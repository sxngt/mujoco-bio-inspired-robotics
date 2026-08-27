# labs/ch12-reward-shaping/04_curriculum.py
"""12.4 Curriculum: from easy to hard.

In an environment with push disturbances (up to 60 N): (a) train 6M steps at 60 N from the start (scratch),
(b) continue from R4 (flat ground, 3M) for 3M steps while ramping 20 → 60 N (first attempt),
(c) the same curriculum over 6M: ramp 20 → 60 N across 3M, then 3M more at 60 N,
(d) beyond the test force: ramp 20 → 80 N across 4M, then 2M at 80 N,
(e) control group: 6M at a weak constant 20 N (disturbance only, no curriculum),
(f) same as (c) but with a 0.15 rad correction range: checks whether the residual range is the bottleneck. Push survival rates and learning curves. Survival rates are saved to out/push_survival.json.
Figures: out/ch12-fig07-curriculum-curves.png, out/ch12-fig08-push-limits.png
Run: uv run python 04_curriculum.py
"""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_lib import OUT, load_policy, run_episode

names = ("R4_full", "C_scratch_push", "C_const20_6M", "C_curriculum_push", "C_curriculum_6M", "C_curriculum_80", "C_curriculum_6M_wide")
M = {n: json.load(open(OUT / f"metrics_{n}.json")) for n in names}
M["baseline"] = json.load(open(OUT / "metrics_baseline.json")); M["ppo_first"] = json.load(open(OUT / "metrics_ppo_first.json"))
print("5-seed mean (flat-ground eval)     speed[m/s]  falls  yaw[°]  fragments/cycle          impact[N]  CoT   push limit[N]")
for n in ("baseline", "ppo_first") + names:
    m = M[n]; print(f"  {n:18s} {m['speed']:8.3f}  {m['fall_rate']:5.0%}  {m['yaw_deg']:+6.0f}  {[round(f, 1) for f in m['fragments']] if m['fragments'] else 'n/a'}   {m['impact']:5.0f}  {m['cot']:.2f}   {m['push_limit']}")


def curve(name, key="rollout/ep_rew_mean"):
    rows = list(csv.DictReader(open(OUT / f"log_{name}" / "progress.csv")))
    x = np.array([float(r["time/total_timesteps"]) for r in rows if r.get(key)]); y = np.array([float(r[key]) for r in rows if r.get(key)])
    return x, y


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
COL = {"R4_full": "#999999", "C_scratch_push": "#d62728", "C_const20_6M": "#ff7f0e", "C_curriculum_push": "#2ca02c", "C_curriculum_6M": "#1f77b4", "C_curriculum_80": "#9467bd", "C_curriculum_6M_wide": "#17becf"}
for n in ("R4_full", "C_scratch_push", "C_const20_6M", "C_curriculum_6M", "C_curriculum_6M_wide"):     # only five curves (the 3M and 80 N runs appear in the survival figure)
    x, y = curve(n); off = 3e6 if n.startswith("C_c") else 0.0
    axes[0].plot((x + off) / 1e6, y, color=COL[n], lw=1.0, label=n + (" (from R4 at 3M)" if off else ""))
    x, y = curve(n, "rollout/ep_len_mean"); axes[1].plot((x + off) / 1e6, y, color=COL[n], lw=1.0, label=n)
axes[0].axvline(6.0, color="#bbbbbb", ls=":", lw=0.8); axes[1].axvline(6.0, color="#bbbbbb", ls=":", lw=0.8)
axes[0].text(6.05, 250, "ramp ends: 60 N held", fontsize=7, color="#666666")
axes[0].set_xlabel("timesteps [M]"); axes[0].set_ylabel("episode return"); axes[0].set_title("(a) return (pushes make it noisy)", fontsize=10)
axes[1].set_xlabel("timesteps [M]"); axes[1].set_ylabel("episode length"); axes[1].set_title("(b) episode length (below 1000 = falls)", fontsize=10)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=min(4, max(1, len(l))), fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT / "ch12-fig07-curriculum-curves.png", dpi=200); plt.close(fig)

# Fine-grained push test: 30 to 90 N, both directions, 3 seeds → survival rate
forces = (30, 40, 50, 60, 70, 80, 90)
print("Push survival rate [%] (per force, ±y two directions × 3 seeds)")
surv = {}
for n in names:
    model, venv, env = load_policy(n); pol = lambda o: model.predict(o, deterministic=True)[0]; row = []
    for F in forces:
        ok = [not run_episode(venv, env, pol, seed=s, push=(3.0, sign * F), max_t=6.0)["fell"] for s in (11, 12, 13) for sign in (1, -1)]
        row.append(100 * np.mean(ok))
    surv[n] = row; venv.close()
    print(f"  {n:18s} " + "  ".join(f"{F}N {r:3.0f}" for F, r in zip(forces, row)))
json.dump({"forces": list(forces), **surv}, open(OUT / "push_survival.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for n in names:
    ax.plot(forces, surv[n], "o-", color=COL[n], label=n)
ax.set_xlabel("side push [N] for 0.2 s while trotting"); ax.set_ylabel("survived [%] (2 directions x 3 seeds)"); ax.set_ylim(-5, 105); ax.set_title("push robustness", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch12-fig08-push-limits.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch12-fig07-curriculum-curves.png", OUT / "ch12-fig08-push-limits.png")
