# Chapter 8. Measuring the Walk: Gait Analysis Lab

This chapter produces `quadbook/analysis.py` (dominant_period, gait_metrics, support_margin, froude, relative_stride, alexander_stride), which the reward design in Chapter 12 uses as is.
01 saves the contact log to `out/logs.npz` and 02, 03, 04, and 05 read it. Always run 01 first.

| Script | Section | Contents |
|---|---|---|
| `01_contact_logging.py` | 8.1 | Contact list (geometry) vs touch sensor (force), thresholds, touchdown impact. Saves logs for trot 2 Hz, trot 1 Hz, and walk 1 Hz |
| `02_gait_diagram.py` | 8.2 | Measured gait diagram next to the commanded diagram (plot_gait_diagram from Chapter 3) |
| `03_gait_metrics.py` | 8.3 | Compute β, T, φ, λ, and v as defined in Section 3.2; how β and φ change over a trot frequency sweep |
| `04_compare_animal.py` | 8.4 | Froude number and relative stride: our trots, dogs, and horses on the Alexander & Jayes dynamic similarity line |
| `05_fall_anatomy.py` | 8.5 | Timeline of the 60 N push fall (stability margin, roll, contacts), walking in place on a 10 degree uphill (foot slip, torque saturation), gait change in the 1 Hz trot |

Run from labs/ with `uv run python ch08-gait-analysis/<script>.py`; to watch the scene, add `--view --speed 0.5` (use `mjpython` on macOS).
