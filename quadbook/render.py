"""Helpers for saving scenes as images without opening a window. Every lab script uses them to leave at least one visualization image.

- snapshots(): captures frames at the given times and saves them side by side as one PNG.
- Without a camera name, MuJoCo's free camera (looking at the model center) is used.
"""

from __future__ import annotations

from pathlib import Path

import os
import numpy as np
from PIL import Image, ImageDraw

import mujoco


def _ensure_framebuffer(model: mujoco.MjModel, width: int, height: int) -> None:
    """Enlarge the model's offscreen buffer (default 640x480) if it is smaller than the requested size."""
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)


def _label_font(size: int):
    """Use the DejaVu Sans that ships with matplotlib (present on every OS); fall back to PIL's default font."""
    try:
        import matplotlib
        from PIL import ImageFont
        path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(path, size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    size = max(14, im.width // 30)                       # 30 px on a 900 px frame: still legible at ebook width
    font = _label_font(size)
    box = d.textbbox((0, 0), text, font=font)
    pad = size // 3
    d.rectangle([8, 8, 8 + (box[2] - box[0]) + 2 * pad, 8 + (box[3] - box[1]) + 2 * pad], fill=(255, 255, 255))
    d.text((8 + pad, 8 + pad - box[1]), text, fill=(20, 20, 20), font=font)
    return np.asarray(im)


def snapshots(model: mujoco.MjModel, data: mujoco.MjData, times, out_path, *, camera: str | None = None,
              step_fn=None, width: int = 900, height: int = 600, labels: bool = True) -> Path:
    """Run step_fn (default mj_step) until data.time reaches each value in times, capturing a frame at each.
    If the first value in times is the current time, only mj_forward is computed before capturing."""
    _ensure_framebuffer(model, width, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    step = step_fn or (lambda: mujoco.mj_step(model, data))
    frames = []
    mujoco.mj_forward(model, data)
    for t in times:
        while data.time < t - 1e-9:
            step()
        if camera:
            renderer.update_scene(data, camera=camera)
        else:
            renderer.update_scene(data)
        f = renderer.render().copy()
        frames.append(_label(f, f"t = {data.time:.2f} s") if labels else f)
    renderer.close()
    strip = np.concatenate(frames, axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(out_path)
    return out_path


def poses(model: mujoco.MjModel, data: mujoco.MjData, qpos_list, out_path, *, camera: str | None = None,
          labels=None, width: int = 700, height: int = 600) -> Path:
    """Feed several qpos in turn (mj_forward only), capture each pose, and stitch them together. Time does not advance."""
    _ensure_framebuffer(model, width, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []
    for i, q in enumerate(qpos_list):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        if camera:
            renderer.update_scene(data, camera=camera)
        else:
            renderer.update_scene(data)
        f = renderer.render().copy()
        frames.append(_label(f, labels[i]) if labels else f)
    renderer.close()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(frames, axis=1)).save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Real-time viewer playback: every lab script uses this to show the same scene in a window with the --view option.
# On macOS it must be run with mjpython:  uv run mjpython <script>.py --view
# ---------------------------------------------------------------------------
import sys
import time


def track_camera(model: mujoco.MjModel, body: str = "torso", distance: float = 1.4, azimuth: float = 135.0,
                 elevation: float = -18.0) -> mujoco.MjvCamera:
    """Camera that follows the torso so a walking robot never leaves the frame (pass it as the camera argument of snapshots and poses)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body(body).id
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def want_view() -> bool:
    """True if --view is on the command line."""
    return "--view" in sys.argv


def view_speed(default: float = 1.0) -> float:
    """Read the playback speed factor, e.g. --speed 0.25 (1.0 = real time, 0.25 = quarter-speed slow motion)."""
    if "--speed" in sys.argv:
        i = sys.argv.index("--speed")
        if i + 1 < len(sys.argv):
            try:
                return float(sys.argv[i + 1])
            except ValueError:
                pass
    return default


def play(model: mujoco.MjModel, data: mujoco.MjData, *, step_fn=None, duration: float | None = None,
         realtime: bool = True, speed: float | None = None, title: str = "", fps: float = 60.0) -> None:
    """Open a viewer window and run step_fn (default mj_step) synchronized to the wall clock.

    - The screen refreshes at fps (default 60); at each frame the physics runs just far enough for simulation time to catch up with "wall clock × speed".
      Unlike syncing on every step, this matches real time exactly.
    - speed: 1.0 is real time, 0.25 is 4x slower slow motion. None uses the command-line --speed value (default 1.0).
    - Ends when the window is closed or duration (simulation time) has elapsed, and prints the actual speed factor at the end.
    """
    try:
        import mujoco.viewer
    except Exception as e:  # pragma: no cover
        print("Could not open the viewer:", e)
        return
    step = step_fn or (lambda: mujoco.mj_step(model, data))
    if speed is None:
        speed = view_speed()
    if title:
        print(f"[viewer] {title}  (close the window to quit, speed x{speed:g})")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            t0 = data.time
            wall0 = time.perf_counter()
            frame = 1.0 / fps
            while viewer.is_running():
                frame_start = time.perf_counter()
                if realtime:
                    target = t0 + (frame_start - wall0) * speed
                    n = 0
                    while data.time < target and n < 20000:
                        step(); n += 1
                else:
                    step()
                viewer.sync()
                if duration is not None and data.time - t0 >= duration:
                    break
                rest = frame - (time.perf_counter() - frame_start)
                if rest > 0:
                    time.sleep(rest)
            wall = time.perf_counter() - wall0
            if wall > 0:
                print(f"[viewer] played {data.time - t0:.2f} s of simulation in {wall:.2f} s of wall time (speed x{(data.time - t0) / wall:.2f})")
    except RuntimeError as e:
        msg = str(e)
        if "mjpython" in msg or "main thread" in msg:
            print("On macOS, run with mjpython to open the viewer:  uv run mjpython <script>.py --view")
        else:
            raise
