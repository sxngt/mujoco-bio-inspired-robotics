# labs/ch11-sb3-training/common.py
"""Chapter 11 shared module: the PPO settings of Table 9-2 (Section 9.2) and VecEnv construction. 03 trains, 04 and 05 read the results."""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from quadbook.env import QuadrupedEnv

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
N_ENVS = 8
PPO_KWARGS = dict(                    # Table 9-2
    n_steps=2048, batch_size=256, n_epochs=10, clip_range=0.2, gamma=0.99, gae_lambda=0.95,
    learning_rate=3e-4, ent_coef=0.0, policy_kwargs=dict(log_std_init=-1.0), device="cpu", seed=0,
)


def make_envs(n_envs=N_ENVS, seed=0, subproc=True, **env_kwargs):
    """n QuadrupedEnv instances wrapped in Monitor, run in parallel. The return value is not normalized yet."""
    cls = SubprocVecEnv if subproc else None
    return make_vec_env(QuadrupedEnv, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=cls)


def normalized(venv, training=True):
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=PPO_KWARGS["gamma"], training=training)


def load_trained(name="ppo_first", n_envs=1, **env_kwargs):
    """Load a trained policy and its normalization statistics and return them with an evaluation env (statistics frozen)."""
    venv = make_envs(n_envs=n_envs, seed=123, subproc=False, **env_kwargs)
    venv = VecNormalize.load(str(OUT / f"{name}_vecnormalize.pkl"), venv)
    venv.training = False; venv.norm_reward = False
    model = PPO.load(str(OUT / f"{name}.zip"), env=venv, device="cpu")
    return model, venv
