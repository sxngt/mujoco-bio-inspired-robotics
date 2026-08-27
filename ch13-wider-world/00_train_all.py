# labs/ch13-wider-world/00_train_all.py
"""Run the three Chapter 13 training configs in order and evaluate them (about 20 min, CPU). Run: cd ch13-wider-world && uv run python 00_train_all.py"""
import json
import sys
from train_lib import CONFIGS, OUT, evaluate, train

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(CONFIGS)):
        if (OUT / f"metrics_{name}.json").exists():
            print(f"{name}: already exists", flush=True); continue
        el = train(name); m = evaluate(name); json.dump(m, open(OUT / f"metrics_{name}.json", "w"), indent=1)
        print(f"{name}: {el/60:.1f} min | speed {m['speed']:.3f} m/s, yaw {m['yaw_deg']:+.0f}°, falls {m['fall_rate']:.0%}, fragments {[round(f,1) for f in m['fragments']] if m['fragments'] else None}, impact {m['impact']:.0f} N, CoT {m['cot']:.2f}, freq {m['freq']:.2f} Hz, L {m['step_length']:.3f} m", flush=True)
