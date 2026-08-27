# labs/ch13-wider-world/02_mjx_benchmark.py
"""13.2 Faster training: the road to MJX and large-scale parallelism.

Load the same quadruped.xml into MJX (JAX) and measure throughput per batch size. The laptop used for this book has no GPU, so this is JAX on CPU.
Comparison: one MuJoCo C engine instance (Chapter 4), 8 SubprocVecEnv workers (Chapter 11).
Figure: out/ch13-fig05-mjx-throughput.png
Run: uv run python 02_mjx_benchmark.py   (requires uv sync --extra mjx)
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from quadbook import model_path
from quadbook.robot import load
from train_lib import OUT

model, data = load(model_path())
print("JAX devices:", jax.devices())

# MuJoCo C: one instance
n = 2000; t0 = time.perf_counter()
for _ in range(n): mujoco.mj_step(model, data)
c_rate = n / (time.perf_counter() - t0)
print(f"MuJoCo (C, 1 instance): {c_rate:,.0f} physics steps/s")

# MJX: put the model on the device and jit the batched step
mx = mjx.put_model(model)
dx0 = mjx.put_data(model, data)
step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
rates = {}
for B in (1, 16, 128, 1024):
    batch = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], B, axis=0), dx0)
    t0 = time.perf_counter(); batch = step(mx, batch); jax.block_until_ready(batch); compile_s = time.perf_counter() - t0
    k = 20; t0 = time.perf_counter()
    for _ in range(k): batch = step(mx, batch)
    jax.block_until_ready(batch); el = time.perf_counter() - t0
    rates[B] = B * k / el
    print(f"MJX batch {B:5d}: compile {compile_s:5.1f} s | {rates[B]:10,.0f} physics steps/s ({1000*el/k:.1f} ms per batch)")

fig, ax = plt.subplots(figsize=(7, 3.4))
labels = ["MuJoCo C x1"] + [f"MJX B={B}" for B in rates]; vals = [c_rate] + list(rates.values())
ax.bar(labels, vals, color=["#4a90d9"] + ["#9467bd"] * len(rates)); ax.set_yscale("log"); ax.set_ylabel("physics steps / s (log)"); ax.set_title("throughput on this laptop CPU: MuJoCo vs MJX (JAX on CPU)", fontsize=10); ax.tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig05-mjx-throughput.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch13-fig05-mjx-throughput.png")
