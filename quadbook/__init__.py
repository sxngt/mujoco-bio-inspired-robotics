"""Lab utilities shared across the whole book.

Each chapter's scripts import the functions defined here.
- assets: path to the robot model (MJCF)
- gait:   foot contact logging and gait diagrams (Chapters 3, 8, 12)
"""

from pathlib import Path

try:  # figure defaults: translucent legends placed automatically (so they do not hide data)
    import matplotlib as _mpl
    _mpl.rcParams["legend.framealpha"] = 0.55
    _mpl.rcParams["legend.loc"] = "best"
except Exception:
    pass

LABS_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = LABS_ROOT / "models"


def model_path(name: str = "quadruped.xml") -> Path:
    """Return the absolute path of an MJCF file under models/."""
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path} (it is built in Chapter 5)")
    return path
