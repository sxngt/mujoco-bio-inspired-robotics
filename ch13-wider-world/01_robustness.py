# labs/ch13-wider-world/01_robustness.py
"""13.1 Uneven terrain and disturbances: robustness as the next assignment.

Put the final policy of Chapter 12 (C_curriculum_6M) in conditions it never saw during training: rough terrain (0.5~3 cm tiles), uphill 5° and 10°, friction 0.4 and 0.15, payload 2 and 4 kg.
Side by side with the open-loop generator of Chapter 7 (action 0). Evaluation only, no training.
Figures: out/ch13-fig01-rough-frames.png, out/ch13-fig02-robustness.png
Run: uv run python 01_robustness.py   (--view --speed 0.5: replay the rough terrain)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import mujoco
from quadbook import model_path
from quadbook.render import play, want_view
from quadbook.robot import reset_stand
from train_lib import OUT, load_policy, run_episode, summarize

ROUGH = str(model_path().parent / "quadruped_rough.xml")
FINAL = "C_curriculum_6M"


def conditions():
    def slope(deg):
        def f(env):
            a = np.radians(deg); env.model.opt.gravity[:] = [-9.81 * np.sin(a), 0.0, -9.81 * np.cos(a)]
        return f
    def friction(mu):
        def f(env): env.model.geom_friction[:, 0] = mu
        return f
    def payload(kg):
        def f(env): env.model.body_mass[env.torso] += kg; mujoco.mj_setConst(env.model, env.data); reset_stand(env.model, env.data)
        return f
    return [("flat", {}, None), ("rough terrain", {"model_file": ROUGH}, None), ("slope 5 deg", {}, slope(5)), ("slope 10 deg", {}, slope(10)),
            ("friction 0.4", {}, friction(0.4)), ("friction 0.15", {}, friction(0.15)), ("payload +2 kg", {}, payload(2.0)), ("payload +4 kg", {}, payload(4.0))]


results = {}
print("5-seed mean                      policy       speed[m/s] falls   yaw[°]  impact[N] CoT")
for cname, env_kw, patch in conditions():
    for pname in ("baseline", FINAL):
        model, venv, env = load_policy(FINAL, **env_kw)
        if patch: patch(env)
        pol = (lambda o: np.zeros((1, 12))) if pname == "baseline" else (lambda o: model.predict(o, deterministic=True)[0])
        runs = [summarize(run_episode(venv, env, pol, seed=s)) for s in (7, 8, 9, 10, 11)]
        agg = {"speed": np.mean([r["speed"] for r in runs]), "fall": np.mean([r["fell"] for r in runs]), "yaw": np.mean([r["yaw_deg"] for r in runs]), "impact": np.mean([r["impact"] for r in runs]), "cot": np.mean([r["cot"] for r in runs])}
        results[(cname, pname)] = agg; venv.close()
        print(f"  {cname:16s} {pname:16s} {agg['speed']:8.3f}  {agg['fall']:5.0%}  {agg['yaw']:+6.0f}  {agg['impact']:5.0f}  {agg['cot']:.2f}")
json.dump({f"{c}|{p}": {k: float(v) for k, v in a.items()} for (c, p), a in results.items()}, open(OUT / "robustness.json", "w"), indent=1)

names = [c for c, _, _ in conditions()]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6)); x = np.arange(len(names))
for ax, key, title in zip(axes, ("speed", "fall", "cot"), ("speed [m/s]", "fall rate", "cost of transport")):
    ax.bar(x - 0.2, [results[(c, "baseline")][key] for c in names], 0.4, color="#999999", label="open-loop CPG (ch7)")
    ax.bar(x + 0.2, [results[(c, FINAL)][key] for c in names], 0.4, color="#17becf", label="final policy (ch12)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=20); ax.set_title(title, fontsize=10)
axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig02-robustness.png", dpi=200); plt.close(fig)

model, venv, env = load_policy(FINAL, model_file=ROUGH)
lg = run_episode(venv, env, lambda o: model.predict(o, deterministic=True)[0], seed=7, frames_at=(2.0, 4.0, 6.0, 8.0), label="rough terrain")
Image.fromarray(np.concatenate(lg["frames"], axis=1)).save(OUT / "ch13-fig01-rough-frames.png")
print("saved", OUT / "ch13-fig01-rough-frames.png", OUT / "ch13-fig02-robustness.png")

if want_view():
    state = {"obs": venv.reset()}
    def _step():
        a = model.predict(state["obs"], deterministic=True)[0]; state["obs"], _, _, _ = venv.step(a)
    play(env.model, env.data, step_fn=_step, duration=10.0, title="final policy, rough terrain (never seen in training)")
