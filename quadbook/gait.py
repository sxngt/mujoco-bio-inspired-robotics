"""Gait diagram utilities (Chapter 3 figures, shared with the measured-gait analysis in Chapters 8 and 12).

- gait_intervals(): builds each leg's stance intervals from the phase table (φ) and duty factor (β) (ideal gait).
- plot_gait_diagram(): draws stance intervals as bars. Row order follows the book's convention (LF, RF, LH, RH).
- contacts_to_intervals(): converts a measured binary contact signal c_i(t) into a list of intervals (used in Chapter 8).
"""

from __future__ import annotations

import numpy as np

LEG_ORDER = ("LF", "RF", "LH", "RH")

# Typical values of the Section 3.3 phase table (LF as reference, φ_LF = 0)
GAIT_PHASES: dict[str, dict[str, float]] = {
    "walk": {"LF": 0.0, "RF": 0.5, "LH": 0.75, "RH": 0.25},
    "trot": {"LF": 0.0, "RF": 0.5, "LH": 0.5, "RH": 0.0},
    "pace": {"LF": 0.0, "RF": 0.5, "LH": 0.0, "RH": 0.5},
    "bound": {"LF": 0.0, "RF": 0.0, "LH": 0.5, "RH": 0.5},
    "gallop": {"LF": 0.0, "RF": 0.1, "LH": 0.5, "RH": 0.6},
}

# Typical duty factor per gait (representative values chosen within the table's ranges)
GAIT_DUTY: dict[str, float] = {"walk": 0.7, "trot": 0.45, "pace": 0.45, "bound": 0.4, "gallop": 0.35}


def gait_intervals(phases: dict[str, float], duty: float | dict[str, float], n_cycles: int = 2):
    """Return each leg's stance intervals [(start, end), ...] in cycle units (0 = first LF touchdown)."""
    out: dict[str, list[tuple[float, float]]] = {}
    for leg in LEG_ORDER:
        beta = duty[leg] if isinstance(duty, dict) else duty
        segs = []
        for k in range(-1, n_cycles + 1):
            start = k + phases[leg]
            end = start + beta
            # clip to the drawing range [0, n_cycles]
            s, e = max(start, 0.0), min(end, float(n_cycles))
            if e > s:
                segs.append((s, e))
        out[leg] = segs
    return out


def contacts_to_intervals(t: np.ndarray, contact: np.ndarray):
    """Convert a binary contact signal (1 = stance) into a list of (start, end) time intervals."""
    c = np.asarray(contact).astype(bool)
    if c.size == 0:
        return []
    edges = np.diff(c.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if c[0]:
        starts.insert(0, 0)
    if c[-1]:
        ends.append(len(c) - 1)
    return [(float(t[s]), float(t[e])) for s, e in zip(starts, ends)]


def plot_gait_diagram(ax, intervals: dict[str, list[tuple[float, float]]], *, x_max: float,
                      color="#333333", title: str | None = None, xlabel: str = "gait cycle"):
    """intervals: leg name -> [(start, end), ...]. Row order is LEG_ORDER (LF on top)."""
    n = len(LEG_ORDER)
    for row, leg in enumerate(LEG_ORDER):
        y = n - 1 - row
        for s, e in intervals.get(leg, []):
            ax.broken_barh([(s, e - s)], (y + 0.15, 0.7), color=color)
    ax.set_yticks([n - 1 - r + 0.5 for r in range(n)])
    ax.set_yticklabels(LEG_ORDER)
    ax.set_ylim(0, n)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(xlabel)
    for x in np.arange(0, x_max + 1e-9, 0.5):
        ax.axvline(x, color="#bbbbbb", lw=0.6, ls="--", zorder=0)
    if title:
        ax.set_title(title)
    return ax
