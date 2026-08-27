# Chapter 6. Standing First: Fundamentals of Joint Control

This chapter produces `quadbook/control.py` (JointPD, gravity_compensation, joint_inertia_diag, critical_kd, torso_roll_pitch) and `quadbook.robot.STAND_POSE_BALANCED`, which Chapter 7 onward use as is.

| Script | Section | Contents |
|---|---|---|
| `01_torque_vs_position.py` | 6.1 | Test-rig leg: torque alone stops at the gravity balance angle vs a position actuator. Full-robot landing: impact at kp 40 vs 400 |
| `02_pd_intuition.py` | 6.2 | Simple pendulum: P only / PD / PD + gravity compensation (qfrc_bias). Predict ω, ζ, overshoot from the joint inertia and compare with measurement |
| `03_gain_tuning.py` | 6.3 | Full-robot step response (kp 10, 40, 160, ζ=1), ζ comparison, control rate (500/100/50 Hz) and chatter |
| `04_stand_robust.py` | 6.4 | Load balance (hip 0.9 rad), push limit, effect of feet apart and crouching, why IMU attitude feedback does not work |

Run from labs/ with `uv run python ch06-standing-control/<script>.py`; to watch the scene add `--view --speed 0.5` (`mjpython` on macOS).
