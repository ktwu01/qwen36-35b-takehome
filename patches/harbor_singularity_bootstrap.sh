#!/bin/bash
# Experiment-specific replacement for harbor 0.24.0
# harbor/environments/singularity/bootstrap.sh (installed by scripts/setup_harbor_rt.sh).
#
# Why: seismic18 has no Docker and no /etc/subuid entry, so apptainer --fakeroot runs in a
# root-mapped namespace where apt/dnf cannot install packages. The stock bootstrap needs
# /usr/bin/python3 (for its FastAPI exec server) and tmux (for Terminus-2) inside every task
# image and installs them with apt, which fails here. This version uses a host-built,
# read-only runtime bind-mounted at /opt/harbor-rt (relocatable CPython 3.12 + uvicorn/fastapi
# + static tmux 3.7b). The task image, its Python, the agent and the verifier are unchanged.
# Without /opt/harbor-rt it falls through to the stock script (bootstrap.stock.sh).
export WORKDIR="${1:-/app}"; shift
export HARBOR_STAGING="/staging/env_files"
mkdir -p "$WORKDIR"
export DEBIAN_FRONTEND=noninteractive

RT=/opt/harbor-rt
if [ ! -x "$RT/bin/python3" ]; then
  exec /staging/bootstrap.stock.sh "$WORKDIR" "$@"
fi

export HARBOR_PYTHON="$RT/bin/python3"
export TMUX_TMPDIR="${TMUX_TMPDIR:-/tmp/.harbor-tmux}"
mkdir -p "$TMUX_TMPDIR"
for f in "$HOME/.bashrc" "$HOME/.bash_profile"; do
  [ -f "$f" ] || touch "$f"
  grep -q 'TMUX_TMPDIR' "$f" 2>/dev/null || echo "alias tmux='tmux -S $TMUX_TMPDIR/default'" >> "$f"
done
# Expose the static tmux only when the image has none (writable-tmpfs overlay).
if ! command -v tmux >/dev/null 2>&1; then
  mkdir -p /usr/local/bin && ln -sf "$RT/tools/tmux" /usr/local/bin/tmux
fi

if [ -f "$HARBOR_STAGING/setup.sh" ]; then
  echo "[harbor] Running task setup.sh..." >&2
  source "$HARBOR_STAGING/setup.sh"
fi
exec "$HARBOR_PYTHON" "$@"
