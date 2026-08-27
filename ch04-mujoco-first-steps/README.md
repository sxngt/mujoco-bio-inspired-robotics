# Chapter 4. First Steps with MuJoCo

| Script | Section | Contents |
|---|---|---|
| `01_hello_mujoco.py` | 4.2 | Build a model from an XML string, simulate 1 s with mj_step, check the contact count → `out/ch04-fig01-hello-drop.png` |
| `02_viewer.py` | 4.2 | Open a window with launch_passive. **On macOS run with `uv run mjpython`** |
| `03_mjmodel_mjdata.py` | 4.3 | Division of labor between mjModel (constants) and mjData (state), mj_forward vs mj_step, ctrl → `out/ch04-fig03-two-link-poses.png` |
| `04_timestep_contact.py` | 4.4 | Energy blow-up vs timestep and integrator, reading mjContact, solref and penetration depth → `out/ch04-fig04-double-pendulum.png`, `out/ch04-fig05-timestep-stability.png` |
| `05_python_control.py` | 4.5 | motor + Python PD vs position actuator → `out/ch04-fig06-pendulum-control.png`, `out/ch04-fig07-control-response.png` |
| `06_render_offscreen.py` | 4.2 | Save images without a window using mujoco.Renderer → `out/ch04-fig02-first-scene.png` |

Run from labs/ with `uv run python ch04-mujoco-first-steps/<script>.py`.
