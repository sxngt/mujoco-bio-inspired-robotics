"""Constants and small helpers for the book's quadruped robot (models/quadruped.xml).

Every chapter from Chapter 6 on shares the names and ordering defined here. Joint order matches the actuator order in the XML.
"""

from __future__ import annotations

import numpy as np

import mujoco

LEGS = ("LF", "RF", "LH", "RH")                      # same notation as Chapter 3
JOINTS_PER_LEG = ("abduction", "hip", "knee")
JOINT_NAMES = [f"{leg}_{j}" for leg in LEGS for j in JOINTS_PER_LEG]   # 12 joints, same order as the actuators
FOOT_GEOMS = [f"{leg}_foot" for leg in LEGS]
TOUCH_SENSORS = [f"{leg}_touch" for leg in LEGS]

# Standing pose (identical to keyframe "stand"). Abduction 0, hip 0.8 rad forward, knee folded to -1.5 rad
STAND_POSE = np.array([0.0, 0.8, -1.5] * 4)
# Standing pose with front/rear load balanced in Chapter 6 (hip 0.9 rad). Used as the default from Chapter 7 on
STAND_POSE_BALANCED = np.array([0.0, 0.9, -1.5] * 4)
NQ_ROOT, NV_ROOT = 7, 6                              # qpos and qvel sizes taken up by the freejoint


def load(model_path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the model and return (model, data) initialized to the 'stand' keyframe."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    reset_stand(model, data)
    return model, data


def reset_stand(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)


def joint_qpos(data: mujoco.MjData) -> np.ndarray:
    """The 12 joint angles (skips the 7 freejoint entries)."""
    return data.qpos[NQ_ROOT:NQ_ROOT + 12]


def joint_qvel(data: mujoco.MjData) -> np.ndarray:
    """The 12 joint velocities (skips the 6 freejoint entries)."""
    return data.qvel[NV_ROOT:NV_ROOT + 12]


def pd_torque(data: mujoco.MjData, q_target: np.ndarray, kp: float = 40.0, kd: float = 1.0) -> np.ndarray:
    """Joint PD control torque. Covered in detail in Chapter 6; the Chapter 5 checks only borrow it as a tool to hold a pose."""
    return kp * (q_target - joint_qpos(data)) - kd * joint_qvel(data)


def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Whether each of the 4 feet touches the floor (0/1 in LF, RF, LH, RH order). Reads the contact list directly."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS]
    out = np.zeros(4, dtype=int)
    for i in range(data.ncon):
        c = data.contact[i]
        for k, f in enumerate(feet):
            if (c.geom1 == floor and c.geom2 == f) or (c.geom2 == floor and c.geom1 == f):
                out[k] = 1
    return out


def touch(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Values of the 4 foot touch sensors (normal force magnitude, N)."""
    return np.array([data.sensor(n).data[0] for n in TOUCH_SENSORS])
