"""Shared loop for gait experiments (Chapter 7 on). Joint PD runs every physics step; the rhythm generator updates every control_every steps."""

from __future__ import annotations

import numpy as np

import mujoco

from .control import JointPD, critical_kd, joint_inertia_diag, torso_roll_pitch
from .robot import foot_contacts, joint_qpos, load, touch

KP = 40.0          # Conclusion of Chapter 6: joint stiffness while standing
WALK_KP = 80.0     # Conclusion of Section 7.3: doubled while walking to reduce tracking error (together with velocity feedforward)


def standard_pd(model, data, kp: float = KP, zeta: float = 1.0) -> JointPD:
    return JointPD(kp, critical_kd(kp, joint_inertia_diag(model, data), zeta))


def fallen(data, z_min: float = 0.15, tilt_max: float = 0.8) -> bool:
    roll, pitch = torso_roll_pitch(data)
    return bool(data.body("torso").xpos[2] < z_min or abs(roll) > tilt_max or abs(pitch) > tilt_max)


def rollout(model, data, gen, pd, duration: float, *, control_every: int = 5, settle: float = 0.5,
            feedforward: bool = True, on_step=None, log_every: int = 1) -> dict:
    """Call gen.targets(dt) every control_every steps and track the result with PD.

    For the first settle seconds the robot holds its standing pose (gen.targets_at(initial phase), not the generator's phase-0 target), then starts walking.
    on_step(model, data) is the hook for injecting disturbances every step. The returned log holds time, torso position, roll/pitch, foot contacts, and commanded foot height.
    """
    dt_ctrl = model.opt.timestep * control_every
    log = {k: [] for k in ("t", "x", "y", "z", "roll", "pitch", "contact", "touch", "foot_cmd", "q_cmd", "q_err", "tau", "fell")}
    q_cmd, qd_cmd = gen.targets_at(gen.osc.theta), np.zeros(12)
    i = 0
    while data.time < duration:
        if data.time >= settle and i % control_every == 0:
            q_cmd, qd_cmd = gen.targets_with_velocity(dt_ctrl)      # target angles and target angular velocities
        data.ctrl[:] = pd.torque(data, q_cmd, qd_cmd if feedforward else 0.0)
        if on_step:
            on_step(model, data)
        mujoco.mj_step(model, data)
        if i % log_every == 0:
            r, p = torso_roll_pitch(data)
            pos = data.body("torso").xpos
            log["t"].append(data.time); log["x"].append(pos[0]); log["y"].append(pos[1]); log["z"].append(pos[2])
            log["roll"].append(r); log["pitch"].append(p)
            log["contact"].append(foot_contacts(model, data)); log["touch"].append(touch(model, data))
            log["foot_cmd"].append(gen.foot_height_cmd(gen.osc.theta)); log["q_cmd"].append(q_cmd.copy())
            log["q_err"].append(np.abs(q_cmd - joint_qpos(data))); log["tau"].append(np.abs(data.ctrl).max())
            log["fell"].append(fallen(data))
        i += 1
    return {k: np.array(v) for k, v in log.items()}


def summary(log: dict, t_from: float = 2.0) -> dict:
    """Mean speed (x direction), roll/pitch standard deviation, and whether the robot fell, all after t_from."""
    m = log["t"] >= t_from
    if m.sum() < 2:
        return {"speed": 0.0, "roll_std": 0.0, "pitch_std": 0.0, "fell": bool(log["fell"].any()), "height": float(log["z"][-1])}
    speed = (log["x"][m][-1] - log["x"][m][0]) / (log["t"][m][-1] - log["t"][m][0])
    return {"speed": float(speed), "roll_std": float(np.degrees(log["roll"][m].std())),
            "pitch_std": float(np.degrees(log["pitch"][m].std())), "fell": bool(log["fell"].any()),
            "height": float(log["z"][m].mean()), "track_err": float(np.degrees(log["q_err"][m][:, [1, 2, 4, 5, 7, 8, 10, 11]].mean())),
            "tau_max": float(log["tau"][m].max())}
