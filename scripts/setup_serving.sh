#!/bin/bash
# One-time per host: make the venv's pip CUDA 13 toolchain usable by FlashInfer JIT.
set -euo pipefail
V=${VENV:-/var/tmp/$USER/venvs/train}
C=$V/lib/python3.12/site-packages/nvidia/cu13
[ -x "$V/bin/ninja" ] || UV_CACHE_DIR=/var/tmp/$USER/uv-cache "$HOME/.local/bin/uv" pip install -q -p "$V/bin/python" ninja
[ -e "$C/lib/libcudart.so" ] || ln -s "$C/lib/libcudart.so.13" "$C/lib/libcudart.so"
echo "serving toolchain ready: $C"
