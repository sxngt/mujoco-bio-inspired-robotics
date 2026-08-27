# labs/ch10-gym-environment/01_env_interface.py
"""10.1 Understanding the gym.Env interface.

Create a QuadrupedEnv, inspect its observation and action spaces, and pass Gymnasium's env_checker.
An episode with action 0 (no residual) should reproduce the Chapter 7 trot exactly. Compare it with random actions.
Measure the environment's throughput (control steps per second) to estimate the training budget of Chapter 11.
Figures: out/ch10-fig01-step-timeline.png, out/ch10-fig02-zero-vs-random.png
Run: uv run python ch10-gym-environment/01_env_interface.py   (--view --speed 0.5: an action-0 episode)
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import quadbook.render as R
from quadbook.env import QuadrupedEnv
from quadbook.render import play, want_view

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

env = QuadrupedEnv(render_mode="rgb_array")
print("observation_space:", env.observation_space)
print("action_space     :", env.action_space)
print(f"control period dt = {env.dt} s (physics {env.model.opt.timestep} s × {env.control_every}), episode at most {env.max_steps} steps = {env.max_steps*env.dt:.0f} s")
check_env(env, skip_render_check=True)
print("gymnasium env_checker: passed")


def run(env, policy, seed=0, frames_at=()):
    obs, info = env.reset(seed=seed)
    total, frames, n = 0.0, [], 0
    while True:
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        total += r; n += 1
        if any(abs(info["t"] - ft) < env.dt / 2 for ft in frames_at):
            frames.append(R._label(env.render(), f"t = {info['t']:.2f} s  x = {info['x']:+.2f} m"))
        if term or trunc:
            break
    return {"steps": n, "return": total, "x": info["x"], "terminated": term, "truncated": trunc, "frames": frames}


zero = run(env, lambda o: np.zeros(12), frames_at=(2.0, 6.0))
rng = np.random.default_rng(0)
rand = run(env, lambda o: rng.uniform(-1, 1, 12), frames_at=(2.0, 6.0))
for name, res in (("action 0 (no residual = Chapter 7 trot)", zero), ("random action (±0.1 rad)", rand)):
    print(f"{name}: {res['steps']} steps, forward {res['x']:+.2f} m, return {res['return']:.1f}, terminated {res['terminated']}, truncated {res['truncated']}")
Image.fromarray(np.concatenate(zero["frames"] + rand["frames"], axis=1)).save(OUT / "ch10-fig02-zero-vs-random.png")

# Throughput
obs, _ = env.reset(seed=1); t0 = time.perf_counter(); n = 0
while n < 2000:
    obs, r, term, trunc, _ = env.step(np.zeros(12)); n += 1
    if term or trunc: env.reset()
el = time.perf_counter() - t0
print(f"throughput: {n/el:,.0f} control steps/s (simulation time at {n*env.dt/el:.0f}x real time). One million steps takes about {1e6/(n/el)/60:.0f} min")

# Diagram: the time axis of a single step()
fig, ax = plt.subplots(figsize=(9, 2.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
def box(x, y, w, h, text, fc="#eef3fb", ec="#4a6fa5", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=1)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
box(0.01, 0.55, 0.15, 0.3, "action a\n(12, [-1, 1])")
box(0.19, 0.55, 0.2, 0.3, "CPG target q_cpg(t)\n+ 0.1 * a", fc="#f3f7ee", ec="#5a8a3a")
for k in range(5):
    box(0.42 + k * 0.085, 0.55, 0.075, 0.3, f"PD\nmj_step", fc="#f8f0e8", ec="#a0703a", fs=7)
box(0.86, 0.55, 0.13, 0.3, "obs, reward,\nterminated?")
ax.annotate("", (0.19, 0.7), (0.16, 0.7), arrowprops=dict(arrowstyle="->")); ax.annotate("", (0.42, 0.7), (0.39, 0.7), arrowprops=dict(arrowstyle="->")); ax.annotate("", (0.86, 0.7), (0.845, 0.7), arrowprops=dict(arrowstyle="->"))
ax.text(0.63, 0.42, "5 physics substeps x 0.002 s = one control step of 0.01 s (100 Hz)", ha="center", fontsize=8, color="#555555")
ax.text(0.5, 0.12, "env.step(a): the policy acts at 100 Hz, the joint PD and physics run at 500 Hz", ha="center", fontsize=9)
fig.savefig(OUT / "ch10-fig01-step-timeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("saved", OUT / "ch10-fig01-step-timeline.png", OUT / "ch10-fig02-zero-vs-random.png")
env.close()

if want_view():
    env = QuadrupedEnv(); env.reset(seed=0)
    def _step():
        env.step(np.zeros(12))
    play(env.model, env.data, step_fn=lambda: env.step(np.zeros(12)), duration=10.0, title="QuadrupedEnv, action 0: the Chapter 7 trot unchanged")
