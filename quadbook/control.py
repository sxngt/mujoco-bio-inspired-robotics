"""Joint control basics (Chapter 6). Control tools reused unchanged by later chapters.

- JointPD: PD controller with a spring (kp) and a damper (kd) per joint. torque = kp·(target − angle) − kd·velocity
- gravity_compensation: uses the bias force MuJoCo already computed (gravity + centrifugal + Coriolis) directly as feedforward torque
- joint_inertia_diag: diagonal of the joint-space inertia matrix at the current pose. Used to derive kd from kp
- critical_kd: kd for damping ratio ζ, kd = 2·ζ·sqrt(kp·I)
- torso_roll_pitch: roll and pitch from the torso orientation quaternion (for measurement and analysis)
"""

from __future__ import annotations

import numpy as np

import mujoco

from .robot import NV_ROOT, joint_qpos, joint_qvel


class JointPD:
    """Joint-space PD controller. kp and kd are scalars or arrays of length n_joints."""

    def __init__(self, kp, kd, n_joints: int = 12):
        self.kp = np.broadcast_to(np.asarray(kp, dtype=float), (n_joints,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=float), (n_joints,)).copy()

    def torque(self, data: mujoco.MjData, q_target: np.ndarray, qd_target=0.0) -> np.ndarray:
        """Sum of the spring term kp·(q_t − q) and the damper term kd·(qd_t − qd)."""
        return self.kp * (q_target - joint_qpos(data)) + self.kd * (qd_target - joint_qvel(data))


def gravity_compensation(data: mujoco.MjData) -> np.ndarray:
    """Bias force c(q, qd) for the 12 joints. At rest it is pure gravity torque.
    It is computed every step inside mj_step, so this effectively reads the previous step's value; for a standing robot the difference is negligible."""
    return data.qfrc_bias[NV_ROOT:NV_ROOT + 12].copy()


def joint_inertia_diag(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Diagonal of the joint-space inertia matrix M(q) at the current pose (12 joints). Includes armature."""
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)        # MuJoCo 3.12 signature: (model, data, dst). Expands the sparse matrix into a dense one
    return np.diag(M)[NV_ROOT:NV_ROOT + 12].copy()


def critical_kd(kp, inertia, zeta: float = 1.0) -> np.ndarray:
    """kd that gives damping ratio ζ in the second-order system τ = kp·e − kd·ė. ζ = 1 is critical damping (fastest convergence without overshoot)."""
    return 2.0 * zeta * np.sqrt(np.asarray(kp, dtype=float) * np.asarray(inertia, dtype=float))


def torso_roll_pitch(data: mujoco.MjData) -> tuple[float, float]:
    """Return roll (about the x axis) and pitch (about the y axis) in rad from the torso orientation quaternion (w, x, y, z)."""
    w, x, y, z = data.qpos[3:7]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)
