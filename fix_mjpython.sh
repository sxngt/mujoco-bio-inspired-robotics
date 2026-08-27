#!/usr/bin/env bash
# Run this when mjpython cannot find libpython3.12.dylib on macOS with a uv-managed Python.
# Symptom: "Library not loaded: @rpath/libpython3.12.dylib"
# Cause: uv's Python keeps libpython in its own lib/ folder, which is not on the mjpython launcher's search path.
# Fix: create a symlink at .venv/libpython3.12.dylib, one of the locations mjpython searches.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(readlink -f .venv/bin/python)"
LIB="$(dirname "$(dirname "$PY")")/lib/libpython3.12.dylib"
[ -f "$LIB" ] || { echo "libpython not found: $LIB"; exit 1; }
ln -sf "$LIB" .venv/libpython3.12.dylib
echo "linked: .venv/libpython3.12.dylib -> $LIB"
uv run mjpython -c "import mujoco; print('mjpython OK', mujoco.__version__)"
