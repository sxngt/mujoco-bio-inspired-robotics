# labs/ch13-wider-world/train_lib.py
"""Shared by Chapter 13: a copy of the Chapter 12 train_lib (no imports across chapter folders) holding the three Chapter 13 training configs and evaluation helpers."""

import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from quadbook.analysis import gait_metrics, support_margin
from quadbook.env import QuadrupedEnv
from quadbook.gait import contacts_to_intervals
from quadbook.robot import touch

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CH12 = Path(__file__).resolve().parent.parent / "ch12-reward-shaping" / "out"
PPO_KWARGS = dict(n_steps=2048, batch_size=256, n_epochs=10, clip_range=0.2, gamma=0.99, gae_lambda=0.95,
                  learning_rate=3e-4, ent_coef=0.0, policy_kwargs=dict(log_std_init=-1.0), device="cpu", seed=0, target_kl=0.03)
W_FULL = dict(forward=0.0, track=1.0, heading=1.0, lateral=0.5, attitude=0.5, energy=0.002, contact_match=0.5, impact=0.005, action_rate=0.02)
CONFIGS = {
    # Section 13.3: remove the body linear velocity (not measurable on real hardware) from the observation and train 3M from scratch (compare: R4_full of Chapter 12, same reward, same budget)
    "A_no_linvel": dict(weights=W_FULL, steps=3_000_000, env=dict(use_lin_vel=False)),
    # Section 13.1: target 0.8 m/s. Residual only (12-dim) vs the policy also modulating generator frequency and step length (14-dim)
    "B_residual_08": dict(weights=W_FULL, steps=3_000_000, env=dict(v_target=0.8)),
    "B_modulate_08": dict(weights=W_FULL, steps=3_000_000, env=dict(v_target=0.8, modulate_generator=True)),
}


def make_envs(n_envs=8, seed=0, subproc=True, **env_kwargs):
    return make_vec_env(QuadrupedEnv, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv if subproc else None)


def train(name):
    cfg = CONFIGS[name]; env_kw = dict(reward_weights=cfg["weights"], **cfg.get("env", {}))
    venv = VecNormalize(make_envs(**env_kw), norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    model = PPO("MlpPolicy", venv, verbose=0, **PPO_KWARGS); model.set_logger(configure(str(OUT / f"log_{name}"), ["csv"]))
    t0 = time.perf_counter(); model.learn(total_timesteps=cfg["steps"]); el = time.perf_counter() - t0
    model.save(str(OUT / f"{name}.zip")); venv.save(str(OUT / f"{name}_vecnormalize.pkl")); venv.close()
    return el


def load_policy(name, **env_override):
    """Load a Chapter 13 policy or a Chapter 12 policy (C_curriculum_6M, R4_full). Use env_override to change terrain, noise, etc. for evaluation."""
    if name in CONFIGS:
        src, weights, env_kw = OUT, CONFIGS[name]["weights"], {k: v for k, v in CONFIGS[name].get("env", {}).items()}
    else:
        src, weights, env_kw = CH12, W_FULL, {}
    env_kw.update(env_override)
    raw = make_envs(n_envs=1, seed=123, subproc=False, reward_weights=weights, render_mode="rgb_array", **env_kw)
    venv = VecNormalize.load(str(src / f"{name}_vecnormalize.pkl"), raw); venv.training = False; venv.norm_reward = False
    model = PPO.load(str(src / f"{name}.zip"), env=venv, device="cpu")
    return model, venv, venv.venv.envs[0].unwrapped


def run_episode(venv, env, policy, seed=7, push=None, max_t=9.99, frames_at=(), label=""):
    import quadbook.render as R
    venv.reset(); env.reset(seed=seed); obs = venv.normalize_obs(env._get_obs()[None])
    log = {k: [] for k in ("t", "x", "y", "z", "yaw", "contact", "touch", "forward", "energy", "freq", "step_length")}; frames = []
    while True:
        a = policy(obs)
        env.data.xfrc_applied[env.torso, :] = 0.0
        if push and push[0] <= env.data.time < push[0] + 0.2: env.data.xfrc_applied[env.torso, 1] = push[1]
        obs, r, done, info = venv.step(a if np.ndim(a) == 2 else a[None]); i = info[0]
        for k in ("t", "x", "y", "z", "yaw", "contact", "freq", "step_length"): log[k].append(i[k])
        log["touch"].append(touch(env.model, env.data)); log["forward"].append(i["terms"]["forward"]); log["energy"].append(-i["terms"]["energy"])
        for ft in frames_at:
            if abs(i["t"] - ft) < env.dt / 2: frames.append(R._label(env.render(), f"{label}  t = {i['t']:.2f} s  x = {i['x']:+.2f} m"))
        if done[0] or i["t"] >= max_t - 1e-9: break
    out = {k: np.array(v) for k, v in log.items()}; out["fell"] = bool(i["fell"]); out["frames"] = frames
    return out


def summarize(lg):
    mtr = gait_metrics(lg["t"], lg["contact"], t_from=2.0, x=lg["x"]) if len(lg["t"]) > 300 else None
    speed = float(mtr["speed"]) if mtr else float((lg["x"][-1] - lg["x"][0]) / max(lg["t"][-1] - lg["t"][0], 1e-6))
    impacts = [lg["touch"][(lg["t"] >= s) & (lg["t"] <= e), leg].max() for leg in range(4) for s, e in contacts_to_intervals(lg["t"], lg["contact"][:, leg]) if ((lg["t"] >= s) & (lg["t"] <= e)).sum() > 3]
    return {"speed": speed, "yaw_deg": float(np.degrees(lg["yaw"][-1])), "fell": lg["fell"], "t_end": float(lg["t"][-1]),
            "duty": (mtr["duty"].tolist() if mtr else None), "fragments": (mtr["fragments"].tolist() if mtr else None), "stride": (float(mtr["stride"]) if mtr else None),
            "power": float(lg["energy"].mean()), "cot": float(lg["energy"].mean() / (8.6 * 9.81 * max(speed, 0.02))), "impact": float(np.mean(impacts)) if impacts else 0.0,
            "freq": float(lg["freq"].mean()), "step_length": float(lg["step_length"].mean())}


def evaluate(name, seeds=(7, 8, 9, 10, 11), **env_override):
    model, venv, env = load_policy(name, **env_override)
    policy = lambda o: model.predict(o, deterministic=True)[0]
    runs = [summarize(run_episode(venv, env, policy, seed=s)) for s in seeds]
    agg = {}
    for k in runs[0]:
        vals = [r[k] for r in runs if isinstance(r[k], (int, float, bool)) and r[k] is not None]
        agg[k] = float(np.mean(vals)) if vals and not isinstance(runs[0][k], list) else None
    agg["fall_rate"] = float(np.mean([r["fell"] for r in runs])); agg["name"] = name
    for k in ("duty", "fragments"):
        vals = [r[k] for r in runs if r[k] is not None]; agg[k] = np.mean(vals, axis=0).tolist() if vals else None
    venv.close()
    return agg
