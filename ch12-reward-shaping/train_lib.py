# labs/ch12-reward-shaping/train_lib.py
"""Shared by Chapter 12: train under identical conditions with only the reward weights changed (train), and measure with the same yardstick (evaluate).
The settings are copied from Chapter 11's common.py (no imports across chapter folders), with target_kl added for the long runs."""

import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from quadbook.analysis import gait_metrics, support_margin
from quadbook.env import DEFAULT_WEIGHTS, QuadrupedEnv
from quadbook.gait import contacts_to_intervals
from quadbook.robot import touch

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CH11 = Path(__file__).resolve().parent.parent / "ch11-sb3-training" / "out"
PPO_KWARGS = dict(n_steps=2048, batch_size=256, n_epochs=10, clip_range=0.2, gamma=0.99, gae_lambda=0.95,
                  learning_rate=3e-4, ent_coef=0.0, policy_kwargs=dict(log_std_init=-1.0), device="cpu", seed=0,
                  target_kl=0.03)                       # Section 11.4: cut an update short when it grows too large

# Reward configs (12.1 to 12.4). Terms not listed use DEFAULT_WEIGHTS (0)
W_CH11 = dict(forward=1.0, attitude=0.5, energy=0.002)
W_HEADING = dict(W_CH11, heading=1.0, lateral=0.5)
W_TRACK = dict(forward=0.0, track=1.0, heading=1.0, lateral=0.5, attitude=0.5, energy=0.002)
W_CONTACT = dict(W_TRACK, contact_match=0.5)
W_FULL = dict(W_CONTACT, impact=0.005, action_rate=0.02)
CONFIGS = {
    "R1_heading": dict(weights=W_HEADING, steps=3_000_000),
    "R2_track": dict(weights=W_TRACK, steps=3_000_000),
    "R3_contact": dict(weights=W_CONTACT, steps=3_000_000),
    "R4_full": dict(weights=W_FULL, steps=3_000_000),
    "H1_forward_only": dict(weights=dict(forward=1.0, attitude=0.0, energy=0.0), steps=2_000_000),
    "H2_alive_energy": dict(weights=dict(forward=0.0, alive=1.0, energy=0.01, attitude=0.5), steps=2_000_000),
    "H3_impact_heavy": dict(weights=dict(W_CH11, impact=0.05), steps=2_000_000),
    "H4_contact_heavy": dict(weights=dict(W_CH11, contact_match=5.0), steps=2_000_000),
    "C_scratch_push": dict(weights=W_FULL, steps=6_000_000, env=dict(push_max=60.0)),
    "C_curriculum_push": dict(weights=W_FULL, steps=3_000_000, env=dict(push_max=20.0), init_from="R4_full", push_ramp=(20.0, 60.0, 3_000_000)),
    # Section 12.4: 6M at a weak constant disturbance (20 N). A condition created by a bug in the first curriculum code; kept as a control group
    "C_const20_6M": dict(weights=W_FULL, steps=6_000_000, env=dict(push_max=20.0), init_from="R4_full"),
    # Section 12.4, second attempt: match the budget to scratch (6M) and give time at the maximum force
    "C_curriculum_6M": dict(weights=W_FULL, steps=6_000_000, env=dict(push_max=20.0), init_from="R4_full", push_ramp=(20.0, 60.0, 3_000_000)),
    # Ramp beyond the test force (60 N): 20 → 80 N across 4M, then 2M at 80 N
    "C_curriculum_80": dict(weights=W_FULL, steps=6_000_000, env=dict(push_max=20.0), init_from="R4_full", push_ramp=(20.0, 80.0, 4_000_000)),
    # Check whether the residual range is the bottleneck: the same 6M curriculum with only the correction range widened 0.1 → 0.15 rad
    "C_curriculum_6M_wide": dict(weights=W_FULL, steps=6_000_000, env=dict(push_max=20.0, action_scale=0.15), init_from="R4_full", push_ramp=(20.0, 60.0, 3_000_000)),
}


class PushRamp(BaseCallback):
    """Curriculum: raise the maximum push force from lo to hi over ramp_steps, then hold at hi (updated every 500k steps)."""
    def __init__(self, lo, hi, ramp_steps):
        super().__init__(); self.lo, self.hi, self.total = lo, hi, ramp_steps; self.next = 0
    def _on_step(self):
        if self.num_timesteps >= self.next:
            frac = min(1.0, self.num_timesteps / self.total)
            self.training_env.env_method("set_push_max", self.lo + frac * (self.hi - self.lo))   # write to the real environments, not the wrapper
            self.next += 500_000
        return True


def make_envs(n_envs=8, seed=0, subproc=True, **env_kwargs):
    return make_vec_env(QuadrupedEnv, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv if subproc else None)


def train(name):
    cfg = CONFIGS[name]; env_kw = dict(reward_weights=cfg["weights"], **cfg.get("env", {}))
    raw = make_envs(**env_kw)
    if cfg.get("init_from"):
        venv = VecNormalize.load(str(OUT / f"{cfg['init_from']}_vecnormalize.pkl"), raw); venv.training = True; venv.norm_reward = True
        model = PPO.load(str(OUT / f"{cfg['init_from']}.zip"), env=venv, device="cpu")
        model.set_logger(configure(str(OUT / f"log_{name}"), ["csv"]))
    else:
        venv = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO("MlpPolicy", venv, verbose=0, **PPO_KWARGS)
        model.set_logger(configure(str(OUT / f"log_{name}"), ["csv"]))
    cb = PushRamp(*cfg["push_ramp"]) if cfg.get("push_ramp") else None
    t0 = time.perf_counter(); model.learn(total_timesteps=cfg["steps"], callback=cb, reset_num_timesteps=True); el = time.perf_counter() - t0
    model.save(str(OUT / f"{name}.zip")); venv.save(str(OUT / f"{name}_vecnormalize.pkl")); venv.close()
    return el


def load_policy(name):
    """(model, venv, env). If name is 'ppo_first', loads the Chapter 11 output."""
    src = CH11 if name == "ppo_first" else OUT
    weights = W_CH11 if name == "ppo_first" else CONFIGS[name]["weights"]
    extra = {k: v for k, v in (CONFIGS[name].get("env", {}) if name != "ppo_first" else {}).items() if k != "push_max"}   # evaluate without pushes; keep action_scale etc. as in training
    raw = make_envs(n_envs=1, seed=123, subproc=False, reward_weights=weights, render_mode="rgb_array", **extra)
    venv = VecNormalize.load(str(src / f"{name}_vecnormalize.pkl"), raw); venv.training = False; venv.norm_reward = False
    model = PPO.load(str(src / f"{name}.zip"), env=venv, device="cpu")
    return model, venv, venv.venv.envs[0].unwrapped


def run_episode(venv, env, policy, seed=7, push=None, max_t=9.99, frames_at=(), label=""):
    import quadbook.render as R
    venv.reset(); env.reset(seed=seed); obs = venv.normalize_obs(env._get_obs()[None])
    log = {k: [] for k in ("t", "x", "y", "yaw", "contact", "touch", "delta", "margin", "forward", "energy", "terms")}; frames = []
    while True:
        a = policy(obs)
        env.data.xfrc_applied[env.torso, :] = 0.0
        if push and push[0] <= env.data.time < push[0] + 0.2: env.data.xfrc_applied[env.torso, 1] = push[1]
        obs, r, done, info = venv.step(a if np.ndim(a) == 2 else a[None]); i = info[0]
        log["t"].append(i["t"]); log["x"].append(i["x"]); log["y"].append(i["y"]); log["yaw"].append(i["yaw"]); log["contact"].append(i["contact"])
        log["touch"].append(touch(env.model, env.data)); log["delta"].append(env.action_scale * np.clip(np.asarray(a).ravel(), -1, 1))
        log["margin"].append(support_margin(env.model, env.data)[0]); log["forward"].append(i["terms"]["forward"]); log["energy"].append(-i["terms"]["energy"]); log["terms"].append(i["terms"])
        for ft in frames_at:
            if abs(i["t"] - ft) < env.dt / 2: frames.append(R._label(env.render(), f"{label}  t = {i['t']:.2f} s  x = {i['x']:+.2f} m"))
        if done[0] or i["t"] >= max_t - 1e-9: break
    out = {k: (np.array(v) if k != "terms" else v) for k, v in log.items()}; out["fell"] = bool(i["fell"]); out["frames"] = frames
    return out


def summarize(lg):
    mtr = gait_metrics(lg["t"], lg["contact"], t_from=2.0, x=lg["x"]) if len(lg["t"]) > 300 else None
    impacts = []
    for leg in range(4):
        for s, e in contacts_to_intervals(lg["t"], lg["contact"][:, leg]):
            m = (lg["t"] >= s) & (lg["t"] <= e)
            if m.sum() > 3: impacts.append(lg["touch"][m, leg].max())
    speed = float(mtr["speed"]) if mtr else float((lg["x"][-1] - lg["x"][0]) / max(lg["t"][-1] - lg["t"][0], 1e-6))
    return {"speed": speed, "yaw_deg": float(np.degrees(lg["yaw"][-1])), "y": float(lg["y"][-1]), "fell": lg["fell"], "t_end": float(lg["t"][-1]),
            "duty": (mtr["duty"].tolist() if mtr else None), "fragments": (mtr["fragments"].tolist() if mtr else None), "phase": (mtr["phase"].tolist() if mtr else None),
            "power": float(lg["energy"].mean()), "cot": float(lg["energy"].mean() / (8.6 * 9.81 * max(speed, 0.02))), "impact": float(np.mean(impacts)) if impacts else 0.0,
            "margin": float(lg["margin"][lg["t"] > 2].mean()) if (lg["t"] > 2).any() else 0.0, "delta_deg": float(np.degrees(np.abs(lg["delta"]).mean())),
            "contact_match": float(np.mean([t["contact_match"] for t in lg["terms"]])), "height_mean": None}


def evaluate(name, push_forces=(20, 40, 60, 80, 100, 120)):
    """Gait metrics averaged over 5 seeds + push limit. Saved to out/metrics_<name>.json."""
    model, venv, env = load_policy(name)
    policy = (lambda o: np.zeros((1, 12))) if name == "baseline" else (lambda o: model.predict(o, deterministic=True)[0])
    runs = [summarize(run_episode(venv, env, policy, seed=s)) for s in (7, 8, 9, 10, 11)]
    agg = {k: (float(np.mean([r[k] for r in runs])) if isinstance(runs[0][k], (int, float, bool)) and runs[0][k] is not None else None) for k in runs[0]}
    agg["fall_rate"] = float(np.mean([r["fell"] for r in runs]))
    for k in ("duty", "fragments", "phase"):
        vals = [r[k] for r in runs if r[k] is not None]; agg[k] = np.mean(vals, axis=0).tolist() if vals else None
    survived = 0
    for F in push_forces:
        if run_episode(venv, env, policy, seed=11, push=(3.0, F), max_t=6.0)["fell"]: break
        survived = F
    agg["push_limit"] = survived; agg["name"] = name
    json.dump(agg, open(OUT / f"metrics_{name}.json", "w"), indent=1)
    venv.close()
    return agg


def evaluate_baseline():
    raw = make_envs(n_envs=1, seed=123, subproc=False, reward_weights=W_CH11, render_mode="rgb_array")
    venv = VecNormalize(raw, norm_obs=False, norm_reward=False, training=False); env = raw.envs[0].unwrapped
    policy = lambda o: np.zeros((1, 12))
    runs = [summarize(run_episode(venv, env, policy, seed=s)) for s in (7, 8, 9, 10, 11)]
    agg = {k: (float(np.mean([r[k] for r in runs])) if isinstance(runs[0][k], (int, float, bool)) and runs[0][k] is not None else None) for k in runs[0]}
    agg["fall_rate"] = float(np.mean([r["fell"] for r in runs]))
    for k in ("duty", "fragments", "phase"):
        agg[k] = np.mean([r[k] for r in runs if r[k] is not None], axis=0).tolist()
    survived = 0
    for F in (20, 40, 60, 80, 100, 120):
        if run_episode(venv, env, policy, seed=11, push=(3.0, F), max_t=6.0)["fell"]: break
        survived = F
    agg["push_limit"] = survived; agg["name"] = "baseline"
    json.dump(agg, open(OUT / "metrics_baseline.json", "w"), indent=1)
    return agg
