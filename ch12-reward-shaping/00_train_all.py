# labs/ch12-reward-shaping/00_train_all.py
"""Run every Chapter 12 training in order and evaluate each right away (about 75 min, CPU). Existing results are skipped.
Run: cd ch12-reward-shaping && uv run python 00_train_all.py [name ...]
"""

import sys
from pathlib import Path

from train_lib import CONFIGS, OUT, evaluate, evaluate_baseline, train

if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    if not (OUT / "metrics_baseline.json").exists():
        m = evaluate_baseline(); print(f"baseline: speed {m['speed']:.3f}, push limit {m['push_limit']} N", flush=True)
    if not (OUT / "metrics_ppo_first.json").exists():
        m = evaluate("ppo_first"); print(f"ppo_first: speed {m['speed']:.3f}, yaw {m['yaw_deg']:+.0f}°, push limit {m['push_limit']} N", flush=True)
    for name in names:
        if (OUT / f"metrics_{name}.json").exists():
            print(f"{name}: already exists, skipping", flush=True); continue
        el = train(name)
        m = evaluate(name)
        print(f"{name}: {el/60:.1f} min | speed {m['speed']:.3f} m/s, yaw {m['yaw_deg']:+.0f}°, fragments {[round(f, 1) for f in m['fragments']] if m['fragments'] else None}, impact {m['impact']:.0f} N, CoT {m['cot']:.2f}, falls {m['fall_rate']:.0%}, push limit {m['push_limit']} N", flush=True)
