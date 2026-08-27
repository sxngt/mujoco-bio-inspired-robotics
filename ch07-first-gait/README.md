# Chapter 7. First Gait: Walking by Hard-Coding

This chapter produces `quadbook/cpg.py` (LegKinematics, FootTrajectory, PhaseOscillators, GaitGenerator) and `quadbook/sim.py` (standard_pd, rollout, summary, WALK_KP), which Chapter 8 onward uses as is.

| Script | Section | Contents |
|---|---|---|
| `01_foot_trajectory.py` | 7.1 | Foot path traced by joint sine waves vs a designed foot trajectory (stance line + swing arc), inverse kinematics, FK check |
| `02_cpg_oscillator.py` | 7.2 | Coupled phase oscillators: randomly started phases converge to the trot phase table and recover after a disturbance. Oscillator → trajectory → IK pipeline |
| `03_trot.py` | 7.3 | Trot at 2 Hz. Three stages, kp 40 → velocity feedforward → kp 80 (24% → 60% → 91%) |
| `04_walk.py` | 7.4 | Lateral-sequence walk (duty 0.75, 1 Hz) compared with trot |
| `05_param_sweep.py` | 7.5 | Frequency × step length, step height, phase tables (trot·pace·bound·walk·pronk) |
| `06_open_loop_limits.py` | 7.6 | Commanded vs actual contact, side pushes while walking, slope (tilted gravity), 2 kg payload, friction |

Run from labs/ with `uv run python ch07-first-gait/<script>.py`; to watch the scene add `--view --speed 0.5` (`mjpython` on macOS).
