# Chapter 5. Building My Quadruped Robot: MJCF Modeling

The outputs are `labs/models/quadruped.xml` (the main robot of this book) and `quadbook/robot.py` (names, ordering, standing pose, helpers); every chapter from Chapter 6 on reads these two.

| Script | Section | Contents |
|---|---|---|
| `01_torso_only.py` | 5.1 | Model v0: a single torso. Checks the tree, nq/nv, and automatic inertia computation → `out/ch05-fig01-torso-drop.png` |
| `02_one_leg.py` | 5.2 | Model v1: one leg fixed to a test stand, joint ranges and foot workspace → `out/ch05-fig02-one-leg-poses.png`, `out/ch05-fig03-leg-workspace.png` |
| `03_inspect_quadruped.py` | 5.3 | Inspecting the finished model (mass, symmetry, foot positions) and rendering the standing pose → `out/ch05-fig04-quadruped-stand.png` |
| `04_sensors.py` | 5.4 | Reading the IMU and foot touch sensors during a drop → `out/ch05-fig05-drop-frames.png`, `out/ch05-fig06-sensors-drop.png` |
| `05_validate.py` | 5.5 | Drop (with and without control), push, suspicion checklist → `out/ch05-fig07-validation-frames.png`, `out/ch05-fig08-validation.png` |

Run from labs/ with `uv run python ch05-mjcf-modeling/<script>.py`.
