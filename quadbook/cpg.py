"""Rhythm generation and foot trajectories (Chapter 7). From sine waves to a CPG: the gait generator reused unchanged by later chapters.

- LegKinematics: forward kinematics (FK) and inverse kinematics (IK) of a sagittal two-link leg (thigh 0.2 m, calf 0.2 m).
- FootTrajectory: phase φ ∈ [0, 1) → foot position (x, z). Stance is a straight line pushing backward along the ground, swing is an arc carrying the foot forward.
- PhaseOscillators: coupled phase oscillators. Each leg has one phase, and the coupling term pulls the phase differences toward the target phase table.
- GaitGenerator: combines the three above, time → 12 joint targets (abduction joints are 0).
"""

from __future__ import annotations

import numpy as np

from .gait import GAIT_PHASES, LEG_ORDER
from .robot import STAND_POSE_BALANCED

THIGH = 0.2   # thigh length in models/quadruped.xml
CALF = 0.2    # calf length (the foot sphere sits at the end of the calf)


class LegKinematics:
    """Sagittal-plane coordinates with the hip joint at the origin. x is forward (+), z is up (+). A positive hip angle moves the foot backward (model convention)."""

    def __init__(self, l1: float = THIGH, l2: float = CALF):
        self.l1, self.l2 = l1, l2

    def forward(self, hip: float, knee: float) -> tuple[float, float]:
        """Joint angles → foot position (x, z)."""
        x = -(self.l1 * np.sin(hip) + self.l2 * np.sin(hip + knee))
        z = -(self.l1 * np.cos(hip) + self.l2 * np.cos(hip + knee))
        return float(x), float(z)

    def inverse(self, x: float, z: float) -> tuple[float, float]:
        """Foot position (x, z) → joint angles (hip, knee). Picks the knee-bent-backward solution (knee < 0)."""
        u, v = -x, -z                                     # standard two-link coordinates measured from the downward (−z) axis
        r2 = u * u + v * v
        c = (r2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        knee = -np.arccos(np.clip(c, -1.0, 1.0))          # unreachable points are clipped to the nearest reachable one
        hip = np.arctan2(u, v) - np.arctan2(self.l2 * np.sin(knee), self.l1 + self.l2 * np.cos(knee))
        return float(hip), float(knee)


class FootTrajectory:
    """Builds a foot trajectory from a single phase.

    φ < duty  : stance. The foot moves backward in a straight line from x0 + L/2 to x0 − L/2 (the body moves forward by the same amount).
    φ ≥ duty  : swing. The foot returns forward along an arc of height h.
    """

    def __init__(self, x0: float, z0: float, duty: float = 0.5, step_length: float = 0.08, step_height: float = 0.04):
        self.x0, self.z0, self.duty = x0, z0, duty
        self.L, self.h = step_length, step_height

    def __call__(self, phase: float) -> tuple[float, float]:
        phase = phase % 1.0
        if phase < self.duty:
            s = phase / self.duty                         # front to back as s goes 0 → 1
            return self.x0 + self.L * (0.5 - s), self.z0
        s = (phase - self.duty) / (1.0 - self.duty)       # back to front as s goes 0 → 1
        x = self.x0 + self.L * (-0.5 + 0.5 * (1 - np.cos(np.pi * s)))   # zero velocity at both ends (lifts off and lands smoothly)
        z = self.z0 + self.h * np.sin(np.pi * s)
        return float(x), float(z)


class PhaseOscillators:
    """Phases θ_i of the 4 legs (cycle units, [0, 1)).

    dθ_i/dt = f + (k / 2π) · Σ_j sin(2π · (θ_j − θ_i − Δ_ij)),   Δ_ij = φ_j − φ_i (difference in the target phase table)
    With coupling k = 0 each leg runs at f on its own; with k > 0 the phase differences are pulled toward the target.
    """

    def __init__(self, freq: float, offsets: dict[str, float], coupling: float = 2.0, init=None):
        self.freq, self.k = freq, coupling
        self.target = np.array([offsets[leg] for leg in LEG_ORDER])
        self.theta = self.target.copy() if init is None else np.asarray(init, dtype=float) % 1.0

    def step(self, dt: float) -> np.ndarray:
        d = self.theta[None, :] - self.theta[:, None]                 # θ_j − θ_i
        delta = self.target[None, :] - self.target[:, None]           # Δ_ij
        coupling = (self.k / (2 * np.pi)) * np.sin(2 * np.pi * (d - delta)).sum(axis=1)
        self.theta = (self.theta + (self.freq + coupling) * dt) % 1.0
        return self.theta


class GaitGenerator:
    """Time → 12 joint targets. Abduction joints are held at 0 (the Chapter 7 gait moves only in the sagittal plane)."""

    def __init__(self, gait: str = "trot", freq: float = 2.0, duty: float = 0.5, step_length: float = 0.08,
                 step_height: float = 0.04, stance_pose=STAND_POSE_BALANCED, coupling: float = 2.0,
                 offsets: dict[str, float] | None = None, init_phase=None):
        self.kin = LegKinematics()
        self.offsets = offsets if offsets is not None else GAIT_PHASES[gait]
        self.osc = PhaseOscillators(freq, self.offsets, coupling, init=init_phase)
        x0, z0 = self.kin.forward(stance_pose[1], stance_pose[2])       # the foot position of the standing pose is the trajectory center
        self.foot = FootTrajectory(x0, z0, duty, step_length, step_height)
        self.duty, self.freq = duty, freq

    def targets_at(self, phases) -> np.ndarray:
        """4 phases → 12 joint targets ([abduction, hip, knee] in LF, RF, LH, RH order)."""
        q = np.zeros(12)
        for i, ph in enumerate(phases):
            x, z = self.foot(ph)
            q[3 * i + 1], q[3 * i + 2] = self.kin.inverse(x, z)
        return q

    def targets(self, dt: float) -> np.ndarray:
        """Advance the phases by dt and return the joint targets."""
        q = self.targets_at(self.osc.step(dt))
        self._last_q = q
        return q

    def targets_with_velocity(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Joint targets and their rate of change (target angular velocity). The latter goes into JointPD's qd_target slot (left empty in Section 6.2)."""
        q_prev = getattr(self, "_last_q", self.targets_at(self.osc.theta))
        q = self.targets(dt)
        return q, (q - q_prev) / dt

    def foot_height_cmd(self, phases) -> np.ndarray:
        """Commanded foot height (z − z0 relative to the trajectory). 0 means stance."""
        return np.array([self.foot(ph)[1] - self.foot.z0 for ph in phases])

    @property
    def ideal_speed(self) -> float:
        """Without slip, body speed = step length / stance time = L · f / β."""
        return self.foot.L * self.freq / self.duty
