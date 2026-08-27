# labs/ch11-sb3-training/05_compare.py
"""11.5 Hard-coded vs corrected gait: measuring again with the tools of Chapter 8.

Walk the baseline (action 0) and the learned policy (deterministic) for 10 s each in the same environment and compare
(1) speed, energy, and reward, (2) the gait_metrics of Chapter 8 (beta, phi, fragment count), stability margin, and touchdown impact, (3) the shape of the correction delta the policy outputs, (4) the top-view path (heading drift), (5) recovery from a lateral push.
Figures: out/ch11-fig07-learned-frames.png, out/ch11-fig08-delta.png, out/ch11-fig09-gait-diagrams.png, out/ch11-fig10-top-view.png
Run: uv run python ch11-sb3-training/05_compare.py   (--view --speed 0.5: replay the learned policy)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import quadbook.render as R
from quadbook.analysis import gait_metrics, support_margin
from quadbook.gait import LEG_ORDER, contacts_to_intervals, plot_gait_diagram
from quadbook.render import play, want_view
from quadbook.robot import touch

from common import OUT, load_trained

model, venv = load_trained("ppo_first", render_mode="rgb_array")
env = venv.venv.envs[0].unwrapped                       # VecNormalize -> DummyVecEnv -> Monitor -> QuadrupedEnv


def episode(policy, seed=7, push=None, frames_at=(), max_t=9.99):   # at 10.0 the last step would read the state after the VecEnv auto-reset
    obs = venv.reset(); env.reset(seed=seed); obs = venv.normalize_obs(env._get_obs()[None])   # reset with the same seed
    log = {k: [] for k in ("t", "x", "contact", "touch", "delta", "phase", "margin", "roll", "forward", "energy", "reward", "y", "yaw")}
    frames = []
    while True:
        a = policy(obs)
        env.data.xfrc_applied[env.torso, :] = 0.0
        if push and push[0] <= env.data.time < push[0] + 0.2:
            env.data.xfrc_applied[env.torso, 1] = push[1]
        obs, r, done, info = venv.step(a[None] if a.ndim == 1 else a)
        i = info[0]
        log["t"].append(i["t"]); log["x"].append(i["x"]); log["contact"].append(i["contact"]); log["touch"].append(touch(env.model, env.data))
        log["delta"].append(env.action_scale * np.clip(np.asarray(a).ravel(), -1, 1)); log["phase"].append(env.gen.osc.theta.copy())
        log["margin"].append(support_margin(env.model, env.data)[0]); log["roll"].append(np.degrees(np.arctan2(2*(env.data.qpos[3]*env.data.qpos[4]+env.data.qpos[5]*env.data.qpos[6]), 1-2*(env.data.qpos[4]**2+env.data.qpos[5]**2))))
        log["forward"].append(i["terms"]["forward"]); log["energy"].append(-i["terms"]["energy"]); log["reward"].append(venv.get_original_reward()[0]); log["y"].append(env.data.body("torso").xpos[1])
        w, qx, qy, qz = env.data.qpos[3:7]; log["yaw"].append(np.degrees(np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))
        for ft in frames_at:
            if abs(i["t"] - ft) < env.dt / 2: frames.append(R._label(env.render(), f"learned policy  t = {i['t']:.2f} s  x = {i['x']:+.2f} m"))
        if done[0] or i["t"] >= max_t - 1e-9:
            break
    out = {k: np.array(v) for k, v in log.items()}; out["fell"] = bool(i["fell"]); out["frames"] = frames
    return out


baseline = lambda o: np.zeros((1, 12))
learned = lambda o: model.predict(o, deterministic=True)[0]

print("(1) Same environment, 10 s (seed 7)")
res = {}
for name, pol in (("baseline (a = 0)", baseline), ("learned (deterministic)", learned)):
    lg = episode(pol, frames_at=(2.0, 2.125, 2.25, 2.375) if name.startswith("learned") else ())
    res[name] = lg
    mtr = gait_metrics(lg["t"], lg["contact"], t_from=2.0, x=lg["x"])
    impacts = []
    for leg in range(4):
        f = lg["touch"][:, leg]; c = lg["contact"][:, leg]
        for s, e in contacts_to_intervals(lg["t"], c):
            m = (lg["t"] >= s) & (lg["t"] <= e)
            if m.sum() > 3: impacts.append(f[m].max())
    lg["metrics"] = mtr; lg["impact"] = float(np.mean(impacts)); lg["cot"] = float(lg["energy"].mean() / (8.6 * 9.81 * max(mtr["speed"], 1e-6)))
    print(f"  {name:24s}: speed {mtr['speed']:.3f} m/s | reward/step {lg['reward'].mean():+.3f} | power {lg['energy'].mean():.1f} W (CoT {lg['cot']:.2f}) | beta {np.round(mtr['duty'], 2)} | phi {np.round(mtr['phase'], 2)} | fragments/cycle {np.round(mtr['fragments'], 1)} | mean stability margin {100*lg['margin'][lg['t']>2].mean():+.1f} cm | mean peak touchdown force {lg['impact']:.0f} N | mean |delta| {np.degrees(np.abs(lg['delta']).mean()):.2f} deg | fell {lg['fell']}")
lg = res["learned (deterministic)"]
Image.fromarray(np.concatenate(lg["frames"], axis=1)).save(OUT / "ch11-fig07-learned-frames.png")

# (3) Shape of delta: mean magnitude per joint, phase-binned mean for the LF joints
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
names = [f"{l}\n{j}" for l in LEG_ORDER for j in ("abd", "hip", "knee")]
axes[0].bar(range(12), np.degrees(np.abs(lg["delta"]).mean(axis=0)), color="#4a90d9"); axes[0].set_xticks(range(12)); axes[0].set_xticklabels(names, fontsize=7); axes[0].set_ylabel("mean |delta| [deg]"); axes[0].set_title("(a) how much the policy corrects each joint", fontsize=10)
ph = lg["phase"][:, 0]; bins = np.linspace(0, 1, 21); mid = (bins[:-1] + bins[1:]) / 2
for j, lab in ((0, "LF abduction"), (1, "LF hip"), (2, "LF knee")):
    prof = [np.degrees(lg["delta"][(ph >= bins[k]) & (ph < bins[k + 1]), j].mean()) for k in range(20)]
    axes[1].plot(mid, prof, "o-", ms=3, label=lab)
axes[1].axvspan(0, 0.5, color="#dddddd", alpha=0.5); axes[1].text(0.25, axes[1].get_ylim()[1] * 0.9, "stance", ha="center", fontsize=8); axes[1].text(0.75, axes[1].get_ylim()[1] * 0.9, "swing", ha="center", fontsize=8)
axes[1].axhline(0, color="#999999", lw=0.8); axes[1].set_xlabel("LF CPG phase [cycle]"); axes[1].set_ylabel("mean delta [deg]"); axes[1].set_title("(b) correction vs gait phase (LF leg)", fontsize=10); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch11-fig08-delta.png", dpi=200); plt.close(fig)

# (2) Gait diagrams side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, (name, lgx) in zip(axes, res.items()):
    t, c = lgx["t"], lgx["contact"]; T = 0.5
    t0 = [s for s, _ in contacts_to_intervals(t, c[:, 0]) if s > 3.0][0]; m = (t >= t0) & (t <= t0 + 2 * T)
    iv = {leg: [((s - t0) / T, (e - t0) / T) for s, e in contacts_to_intervals(t[m], c[m][:, i])] for i, leg in enumerate(LEG_ORDER)}
    plot_gait_diagram(ax, iv, x_max=2, color="#333333" if "learned" in name else "#999999", title=name + " (measured)")
fig.tight_layout(); fig.savefig(OUT / "ch11-fig09-gait-diagrams.png", dpi=200); plt.close(fig)

# (4) Top-view path: heading drift
print("(4) Position and heading after 10 s (the reward is the forward velocity in the torso frame, so heading is not penalized)")
fig, ax = plt.subplots(figsize=(7, 3.6))
for name, lgx in res.items():
    ax.plot(lgx["x"], lgx["y"], lw=1.5, label=name)
    print(f"  {name:24s}: x {lgx['x'][-1]:+.2f} m, y {lgx['y'][-1]:+.2f} m, yaw {lgx['yaw'][-1]:+.0f} deg")
ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("top view of 10 s: the learned policy walks faster and veers", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "ch11-fig10-top-view.png", dpi=200); plt.close(fig)

# (5) Recovery from a lateral push (table only)
print("(5) Lateral push for 0.2 s at t = 3 s while walking: fell or not")
forces = (20, 40, 60, 80, 100)
for name, pol in (("baseline (a = 0)", baseline), ("learned (deterministic)", learned)):
    fell = [episode(pol, seed=11, push=(3.0, F), max_t=6.0)["fell"] for F in forces]
    print(f"  {name:24s}: " + "  ".join(f"{F} N {'fell' if f else 'held'}" for F, f in zip(forces, fell)))
print("saved", OUT / "ch11-fig07-learned-frames.png", OUT / "ch11-fig08-delta.png", OUT / "ch11-fig09-gait-diagrams.png", OUT / "ch11-fig10-top-view.png")

if want_view():
    obs = venv.reset()
    state = {"obs": obs}
    def _step():
        a = model.predict(state["obs"], deterministic=True)[0]; state["obs"], _, _, _ = venv.step(a)
    play(env.model, env.data, step_fn=_step, duration=10.0, title="Learned policy (PPO 3M steps, deterministic)")
