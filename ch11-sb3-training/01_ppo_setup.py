# labs/ch11-sb3-training/01_ppo_setup.py
"""11.1 SB3 structure and PPO settings.

Build a PPO object and look inside: the policy network (actor and critic), parameter count, the standard deviation of the action distribution (log_std), and how much data one update processes.
Compare the action distribution of the untrained policy with log_std_init at 0 (default) and at -1 (the key to Section 11.3, 'keeping the baseline from collapsing').
Figure: out/ch11-fig01-initial-actions.png
Run: uv run python ch11-sb3-training/01_ppo_setup.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from common import OUT, PPO_KWARGS, make_envs

if __name__ == "__main__":
    venv = make_envs(n_envs=1, subproc=False)
    model = PPO("MlpPolicy", venv, verbose=0, **PPO_KWARGS)
    print("Policy network:"); print(model.policy)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Trainable parameters {n_params:,} | action distribution log_std initial value {model.policy.log_std.data[0].item():.2f} -> std {np.exp(model.policy.log_std.data[0].item()):.2f}")
    print(f"One update: n_steps {PPO_KWARGS['n_steps']} x 8 envs = {PPO_KWARGS['n_steps']*8:,} steps collected -> batch {PPO_KWARGS['batch_size']} x {PPO_KWARGS['n_steps']*8//PPO_KWARGS['batch_size']} minibatches x {PPO_KWARGS['n_epochs']} epochs = {PPO_KWARGS['n_steps']*8//PPO_KWARGS['batch_size']*PPO_KWARGS['n_epochs']} gradient steps")
    print(f"With a 0.01 s control period, one update collects {PPO_KWARGS['n_steps']*8*0.01:.0f} s of simulated experience = about {PPO_KWARGS['n_steps']*8*0.01/10:.0f} 10-second episodes")

    # Action distribution of the untrained policy: log_std_init 0 vs -1
    obs = venv.reset()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, ls in zip(axes, (0.0, -1.0)):
        kw = dict(PPO_KWARGS); kw["policy_kwargs"] = dict(log_std_init=ls)
        m = PPO("MlpPolicy", venv, verbose=0, **kw)
        acts = np.array([m.predict(obs, deterministic=False)[0][0] for _ in range(2000)])
        sat = np.mean(np.abs(acts) >= 0.999)
        print(f"log_std_init {ls:+.0f}: untrained action std {acts.std():.2f}, fraction clipped at |a| = 1 {100*sat:.0f}%, as a joint correction +-{0.1*acts.std():.3f} rad")
        ax.hist(acts.ravel(), bins=40, range=(-1, 1), color="#4a90d9"); ax.set_title(f"log_std_init = {ls:+.0f}: std {acts.std():.2f}, clipped {100*sat:.0f}%", fontsize=10); ax.set_xlabel("action (before x 0.1 rad)")
    axes[0].set_ylabel("count"); fig.tight_layout(); fig.savefig(OUT / "ch11-fig01-initial-actions.png", dpi=200); plt.close(fig)
    print("saved", OUT / "ch11-fig01-initial-actions.png")
