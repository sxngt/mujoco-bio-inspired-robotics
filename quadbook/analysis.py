"""Gait measurement tools (Chapter 8). Computes the Chapter 3 quantities from contact logs and measures the support polygon and stability margin. The Chapter 12 reward design uses these functions as well.

- touchdowns(): extracts touchdown and lift-off times from a binary contact signal.
- gait_metrics(): per-leg duty factor β, period T, relative phase φ with LF as reference, stride length λ (exactly as defined in Section 3.2).
- support_margin(): signed distance between the support polygon formed by the feet in contact and the projected center of mass (Section 3.5).
- froude(), relative_stride(): dynamic similarity comparison (Section 3.4, Alexander & Jayes 1983).
"""

from __future__ import annotations

import numpy as np

from .gait import LEG_ORDER, contacts_to_intervals
from .robot import FOOT_GEOMS, foot_contacts


def touchdowns(t, contact):
    """Binary contact signal → (list of touchdown times, list of lift-off times)."""
    iv = contacts_to_intervals(np.asarray(t), np.asarray(contact))
    return np.array([s for s, _ in iv]), np.array([e for _, e in iv])


def dominant_period(t, contacts, leg=0):
    """Dominant period of one leg's (default LF) contact signal (FFT). Robust to bounces at touchdown and dropouts in mid-stance.
    Summing all four feet would pick up the half period in trot, since the diagonal pairs alternate, so a single leg's signal is used."""
    t = np.asarray(t); sig = np.asarray(contacts, dtype=float)[:, leg]
    sig = sig - sig.mean()
    dt = float(np.mean(np.diff(t)))
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), dt)
    spec[freqs < 0.2] = 0                     # ignore anything below 0.2 Hz (slow drift)
    return float(1.0 / freqs[np.argmax(spec)])


def gait_metrics(t, contacts, t_from=2.0, t_to=None, x=None):
    """t: (N,), contacts: (N, 4) 0/1 (LF RF LH RH). Computed over a whole-cycle window after t_from.

    Returned dict:
      period      dominant period T (FFT), period_td: T measured from LF touchdown gaps (mean of the gaps within ±30% of T)
      duty[4]     β = fraction of time in contact (whole-cycle window)
      phase[4]    relative phase with LF as reference. Uses the touchdown time of the 'longest contact interval' in each cycle and takes the circular mean
      fragments[4] number of contact intervals per cycle (1 means a clean stance, larger means bouncing or dropouts)
      n_cycles, stride (mean distance traveled between LF touchdowns, if x is given), speed
    """
    t = np.asarray(t); contacts = np.asarray(contacts).astype(bool)
    m = t >= t_from
    if t_to is not None:
        m &= t <= t_to
    t, contacts = t[m], contacts[m]
    x = None if x is None else np.asarray(x)[m]
    T = dominant_period(t, contacts)
    n_cycles = int((t[-1] - t[0]) // T)
    if n_cycles < 2:
        return None
    # Representative LF touchdown per cycle: the start of the longest LF contact interval in each cycle window
    def main_touchdowns(col):
        iv = contacts_to_intervals(t, col)
        out = []
        for k in range(n_cycles):
            w0, w1 = t[0] + k * T, t[0] + (k + 1) * T
            cand = [(e - s, s) for s, e in iv if w0 <= s < w1]
            if cand:
                out.append(max(cand)[1])
        return np.array(out)
    lf = main_touchdowns(contacts[:, 0])
    if len(lf) >= 2:
        gaps = np.diff(lf); ok = gaps[(gaps > 0.7 * T) & (gaps < 1.3 * T)]
        T_td = float(ok.mean()) if len(ok) else float("nan")
    else:
        T_td = float("nan")
    win = t <= t[0] + n_cycles * T
    duty = contacts[win].mean(axis=0)
    fragments = np.array([len(contacts_to_intervals(t[win], contacts[win][:, i])) / n_cycles for i in range(4)])
    phase = []
    for i in range(4):
        td = main_touchdowns(contacts[:, i])
        ph = []
        for ti in td:
            prev = lf[lf <= ti]
            if len(prev):
                ph.append(((ti - prev[-1]) / T) % 1.0)
        phase.append(float(np.angle(np.mean(np.exp(2j * np.pi * np.array(ph)))) / (2 * np.pi) % 1.0) if ph else float("nan"))
    out = {"period": T, "period_td": T_td, "duty": np.asarray(duty), "phase": np.array(phase), "fragments": fragments,
           "n_cycles": n_cycles, "lf_touchdowns": lf}
    if x is not None:
        xs = np.interp(lf, t, x) if len(lf) >= 2 else None
        out["stride"] = float(np.diff(xs).mean()) if xs is not None else float("nan")
        out["speed"] = float((x[win][-1] - x[0]) / (t[win][-1] - t[0]))
    return out


def _convex_hull(pts):
    """Convex hull of 2D points (counterclockwise). Fewer than 3 points are returned as is."""
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3:
        return np.array(pts)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _point_segment_distance(p, a, b):
    ab = b - a
    tt = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + tt * ab)))


def support_margin(model, data):
    """(stability margin [m], projected CoM xy, support polygon vertices xy). The margin is + inside the polygon and − outside.
    With 2 feet the support is a 'segment', so the margin is −(distance to the segment); with 1 foot or none, −1 is returned instead of −∞."""
    c = foot_contacts(model, data)
    feet = np.array([data.geom(g).xpos[:2] for g, on in zip(FOOT_GEOMS, c) if on])
    com = data.subtree_com[model.body("torso").id][:2].copy()
    if len(feet) == 0:
        return -1.0, com, feet
    if len(feet) == 1:
        return -float(np.linalg.norm(com - feet[0])), com, feet
    hull = _convex_hull(feet)
    if len(hull) == 2:
        return -_point_segment_distance(com, hull[0], hull[1]), com, hull
    n = len(hull)
    inside = True
    dists = []
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        crossv = (b[0] - a[0]) * (com[1] - a[1]) - (b[1] - a[1]) * (com[0] - a[0])
        if crossv < 0:
            inside = False
        dists.append(_point_segment_distance(com, a, b))
    d = min(dists)
    return (d if inside else -d), com, hull


def froude(v, leg_length, g=9.81):
    """Fr = v² / (g · L)  (Section 3.4)."""
    return float(v * v / (g * leg_length))


def relative_stride(stride, leg_length):
    """Relative stride length λ / L."""
    return float(stride / leg_length)


def alexander_stride(fr):
    """Dynamic similarity regression of Alexander & Jayes (1983): relative stride ≈ 2.4 · Fr^0.34 (cursorial mammals)."""
    return 2.4 * np.power(fr, 0.34)
