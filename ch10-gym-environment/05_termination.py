# labs/ch10-gym-environment/05_termination.py
"""10.5 Termination and reset: what makes an environment learnable.

(a) Episode length distribution of a random policy (0.4 rad) over 50 episodes: terminated (fell) vs truncated (time limit).
(b) The scene at termination: what the height and tilt thresholds catch.
(c) Reset randomization: the initial joint error and CPG phase differ every episode. Reset 20 times with the same policy and look at the distribution of forward distance.
Figures: out/ch10-fig07-episode-lengths.png, out/ch10-fig08-termination-frames.png
Run: uv run python ch10-gym-environment/05_termination.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import quadbook.render as R
from quadbook.control import torso_roll_pitch
from quadbook.env import QuadrupedEnv

OUT = Path(__file__).resolve().parent / "out"

# (a) Length distribution
env = QuadrupedEnv(action_scale=0.4, render_mode="rgb_array")
lengths, kinds, frames = [], [], []
for seed in range(50):
    rng = np.random.default_rng(seed); obs, _ = env.reset(seed=seed); n = 0
    while True:
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 12)); n += 1
        if term or trunc:
            lengths.append(n); kinds.append("terminated" if term else "truncated")
            if term and len(frames) < 3:
                roll, pitch = torso_roll_pitch(env.data)
                frames.append(R._label(env.render(), f"terminated at t = {info['t']:.2f} s  z = {info['z']:.2f} m  roll {np.degrees(roll):+.0f} pitch {np.degrees(pitch):+.0f}"))
            break
lengths, kinds = np.array(lengths), np.array(kinds)
print(f"(a) random 0.4 rad, 50 episodes: terminated {np.sum(kinds=='terminated')}, truncated {np.sum(kinds=='truncated')} | mean length {lengths.mean():.0f} steps ({lengths.mean()*env.dt:.1f} s), min {lengths.min()}, max {lengths.max()}")
if frames:
    Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch10-fig08-termination-frames.png")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(lengths[kinds == "terminated"] * env.dt, bins=20, range=(0, 10), color="#d62728", alpha=0.8, label="terminated (fell)")
ax.hist(lengths[kinds == "truncated"] * env.dt, bins=20, range=(0, 10), color="#4a90d9", alpha=0.8, label="truncated (time limit)")
ax.set_xlabel("episode length [s]"); ax.set_ylabel("episodes"); ax.set_title("random policy (0.4 rad): how episodes end", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch10-fig07-episode-lengths.png", dpi=200); plt.close(fig)

# (c) Reset randomization
env = QuadrupedEnv()
xs, phases = [], []
for seed in range(20):
    obs, _ = env.reset(seed=seed); phases.append(env.gen.osc.theta.copy())
    q0 = env.data.qpos[7:19].copy()
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(12))
        if term or trunc: xs.append(info["x"]); break
phases = np.array(phases)
print(f"(c) action 0, 20 resets: initial CPG phase LF {phases[:,0].min():.2f}~{phases[:,0].max():.2f}, initial joint error ±{env.init_noise} rad | forward in 10 s {np.mean(xs):.2f} m (σ {np.std(xs):.2f}, min {np.min(xs):.2f}, max {np.max(xs):.2f})")
env2 = QuadrupedEnv(random_init_phase=False, init_noise=0.0); xs2 = []
for seed in range(5):
    env2.reset(seed=seed)
    while True:
        obs, r, term, trunc, info = env2.step(np.zeros(12))
        if term or trunc: xs2.append(info["x"]); break
print(f"    5 runs without randomization: forward {np.round(xs2, 3)} (all identical: the simulation is deterministic)")
print("saved", OUT / "ch10-fig07-episode-lengths.png", OUT / "ch10-fig08-termination-frames.png")
