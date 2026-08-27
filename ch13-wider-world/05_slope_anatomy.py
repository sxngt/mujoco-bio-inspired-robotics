# labs/ch13-wider-world/05_slope_anatomy.py
"""13.3 (supplement) The policy on a slope: are the two kinds of 'slope' the same to the policy?

Since Chapter 7, slopes were made by tilting the gravity vector (the floor stays level). To the open-loop generator this equals tilting the floor, but to a policy that reads the IMU it does not.
With tilted gravity the torso stays parallel to the floor, so the observed gravity vector matches flat ground; with a tilted floor the torso tilts with it and the gravity vector changes.
Compare the final policy's speed and yaw, and the x component of the observed gravity vector, on both slopes (5°, 10°).
Figure: out/ch13-fig08-slope-two-ways.png
Run: uv run python 05_slope_anatomy.py
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadbook import model_path
from train_lib import OUT, load_policy, run_episode, summarize

FINAL = "C_curriculum_6M"
base_xml = open(model_path(), encoding="utf-8").read()


def tilted_floor_model(deg):
    """Model file with the floor plane tilted about the y axis. Rotated by −deg so that +x is uphill. Robot start height unchanged (above the origin)."""
    xml = re.sub(r'<geom name="floor" type="plane" size="5 5 0.1" material="grid" contype="1" conaffinity="1"/>',
                 f'<geom name="floor" type="plane" size="5 5 0.1" material="grid" contype="1" conaffinity="1" euler="0 {-np.radians(deg):.5f} 0"/>', base_xml)
    assert xml != base_xml
    p = model_path().parent / f"quadruped_slope{deg}.xml"; open(p, "w", encoding="utf-8").write(xml); return str(p)


def gravity_slope(env, deg):
    a = np.radians(deg); env.model.opt.gravity[:] = [-9.81 * np.sin(a), 0.0, -9.81 * np.cos(a)]


rows = {}
print("final policy, 5-seed mean       speed[m/s] falls   yaw[°]   observed gravity x (flat 0)")
for deg in (0, 5, 10):
    for kind in ("gravity tilted", "floor tilted"):
        if deg == 0 and kind == "floor tilted": continue
        if kind == "gravity tilted":
            model, venv, env = load_policy(FINAL); gravity_slope(env, deg)
        else:
            model, venv, env = load_policy(FINAL, model_file=tilted_floor_model(deg))
        pol = lambda o: model.predict(o, deterministic=True)[0]
        gx, runs = [], []
        for s in (7, 8, 9, 10, 11):
            lg = run_episode(venv, env, pol, seed=s); runs.append(summarize(lg))
            R_ = env._torso_rotation(); gx.append((R_.T @ np.array([0, 0, -1.0]))[0])
        agg = {"speed": np.mean([r["speed"] for r in runs]), "fall": np.mean([r["fell"] for r in runs]), "yaw": np.mean([r["yaw_deg"] for r in runs]), "gx": np.mean(gx)}
        rows[(deg, kind)] = agg; venv.close()
        print(f"  {deg:2d}° {kind:15s} {agg['speed']:8.3f}  {agg['fall']:5.0%}  {agg['yaw']:+6.0f}   {agg['gx']:+.3f}")

# Why does it turn: does the policy correction become left/right asymmetric on a slope (seed 7, mean δ)
print("left/right asymmetry of the policy correction (seed 7, mean δ [°]): left (LF abd hip knee | LH abd hip knee) / right (RF | RH), yaw rate")
for label, env_kw, grav in (("flat", {}, 0), ("gravity tilted 5", {}, 5), ("floor tilted 5", {"model_file": tilted_floor_model(5)}, 0)):
    model, venv, env = load_policy(FINAL, **env_kw)
    if grav: gravity_slope(env, grav)
    venv.reset(); env.reset(seed=7); obs = venv.normalize_obs(env._get_obs()[None]); D, Y, T = [], [], []
    while True:
        a = model.predict(obs, deterministic=True)[0]; obs, r, done, info = venv.step(a); D.append(0.1 * np.clip(a.ravel(), -1, 1)); Y.append(info[0]["yaw"]); T.append(info[0]["t"])
        if done[0] or info[0]["t"] >= 9.99: break
    D = np.degrees(np.array(D)); rate = np.gradient(np.degrees(np.unwrap(Y)), np.array(T))[np.array(T) > 2].mean()
    print(f"  {label:18s}: left {np.round(D[:, [0, 1, 2, 6, 7, 8]].mean(axis=0), 1)} / right {np.round(D[:, [3, 4, 5, 9, 10, 11]].mean(axis=0), 1)} | yaw rate {rate:+.1f}°/s")
    venv.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for kind, c in (("gravity tilted", "#999999"), ("floor tilted", "#17becf")):
    degs = [0, 5, 10]; sp = [rows[(0, "gravity tilted")]["speed"]] + [rows[(d, kind)]["speed"] for d in (5, 10)]; yw = [abs(rows[(0, "gravity tilted")]["yaw"])] + [abs(rows[(d, kind)]["yaw"]) for d in (5, 10)]
    axes[0].plot(degs, sp, "o-", color=c, label=kind); axes[1].plot(degs, yw, "o-", color=c, label=kind)
axes[0].set_xlabel("slope [deg]"); axes[0].set_ylabel("speed [m/s]"); axes[0].set_title("(a) speed on two kinds of slope", fontsize=10); axes[0].legend(fontsize=8)
axes[1].set_xlabel("slope [deg]"); axes[1].set_ylabel("|yaw| after 10 s [deg]"); axes[1].set_title("(b) heading drift", fontsize=10); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig08-slope-two-ways.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch13-fig08-slope-two-ways.png")
