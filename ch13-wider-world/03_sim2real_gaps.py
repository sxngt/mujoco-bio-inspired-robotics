# labs/ch13-wider-world/03_sim2real_gaps.py
"""13.3 Out of the simulator: the reality of sim-to-real.

Mimic three gaps every real robot presents inside the simulator and test the final policy against them.
(a) Observation noise: Gaussian noise with std 0.01~0.1 added to the observation before normalization.
(b) Action latency: the policy output arrives 1~3 control steps (10~30 ms) late.
(c) Missing sensor: compare A_no_linvel, trained with the body linear velocity removed, against R4_full (with linear velocity) on the same scale. The policy without linear velocity also gets terrain and pushes.
Figures: out/ch13-fig06-noise-latency.png, out/ch13-fig07-no-linvel.png
Run: uv run python 03_sim2real_gaps.py   (after 00_train_all.py)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_lib import OUT, evaluate

FINAL = "C_curriculum_6M"
print("(a) observation noise (Gaussian added to the observation before normalization, 5 seeds)")
noise_res = {}
for sd in (0.0, 0.01, 0.03, 0.1):
    m = evaluate(FINAL, obs_noise=sd); noise_res[sd] = m
    print(f"    σ {sd:4.2f}: speed {m['speed']:.3f} m/s, falls {m['fall_rate']:.0%}, yaw {m['yaw_deg']:+.0f}°, impact {m['impact']:.0f} N")
print("(b) action latency (control step = 10 ms)")
lat_res = {}
for k in (0, 1, 2, 3):
    m = evaluate(FINAL, latency_steps=k); lat_res[k] = m
    print(f"    {k} steps ({10*k} ms): speed {m['speed']:.3f} m/s, falls {m['fall_rate']:.0%}, yaw {m['yaw_deg']:+.0f}°, impact {m['impact']:.0f} N")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(list(noise_res), [noise_res[s]["speed"] for s in noise_res], "o-", label="speed [m/s]"); axes[0].plot(list(noise_res), [noise_res[s]["fall_rate"] for s in noise_res], "s--", label="fall rate")
axes[0].set_xlabel("observation noise std"); axes[0].set_title("(a) observation noise", fontsize=10); axes[0].legend(fontsize=8)
axes[1].plot(list(lat_res), [lat_res[k]["speed"] for k in lat_res], "o-", label="speed [m/s]"); axes[1].plot(list(lat_res), [lat_res[k]["fall_rate"] for k in lat_res], "s--", label="fall rate")
axes[1].set_xlabel("action latency [control steps of 10 ms]"); axes[1].set_title("(b) action latency", fontsize=10); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig06-noise-latency.png", dpi=200); plt.close(fig)

print("(c) policy trained without body linear velocity (A_no_linvel) vs with it (R4_full), same reward, same 3M steps")
rows = {}
for name in ("R4_full", "A_no_linvel"):
    base = evaluate(name); rough = evaluate(name, model_file=str(__import__("quadbook").model_path().parent / "quadruped_rough.xml")); lat = evaluate(name, latency_steps=2)
    rows[name] = (base, rough, lat)
    print(f"    {name:12s} flat: speed {base['speed']:.3f}, yaw {base['yaw_deg']:+.0f}°, impact {base['impact']:.0f} N, CoT {base['cot']:.2f} | rough terrain: speed {rough['speed']:.3f}, falls {rough['fall_rate']:.0%} | latency 2 steps: speed {lat['speed']:.3f}, falls {lat['fall_rate']:.0%}")
fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(3)
for k, (name, c) in enumerate((("R4_full", "#8c564b"), ("A_no_linvel", "#e377c2"))):
    ax.bar(x + (k - 0.5) * 0.4, [r["speed"] for r in rows[name]], 0.4, color=c, label=name + (" (with body velocity)" if name == "R4_full" else " (without)"))
ax.set_xticks(x); ax.set_xticklabels(["flat", "rough terrain", "latency 20 ms"]); ax.set_ylabel("speed [m/s]"); ax.set_title("does the policy need the simulator's body velocity?", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig07-no-linvel.png", dpi=200); plt.close(fig)
json.dump({"noise": {str(k): v for k, v in noise_res.items()}, "latency": {str(k): v for k, v in lat_res.items()}, "linvel": {n: [r for r in rows[n]] for n in rows}}, open(OUT / "sim2real.json", "w"), indent=1, default=float)
print("saved", OUT / "ch13-fig06-noise-latency.png", OUT / "ch13-fig07-no-linvel.png")
