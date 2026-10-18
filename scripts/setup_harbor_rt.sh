#!/bin/bash
# Build the read-only runtime mounted into every Terminal-Bench container and install the
# patched singularity bootstrap into the harbor venv. Idempotent.
#   bash scripts/setup_harbor_rt.sh [/var/tmp/kw35262]
set -euo pipefail
BASE=${1:-/var/tmp/$USER}
RT=$BASE/harbor-rt
HV=$BASE/venvs/harbor
TMUX_URL=https://github.com/mjakob-gh/build-static-tmux/releases/download/v3.7b/tmux.linux-amd64.stripped.gz
TMUX_SHA=92ac102a1f9b33b21d891a836b4b5b5c8c5d2eeac4bae4dfd34d565bbed598bf
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.12.15%2B20261013-x86_64-unknown-linux-gnu-install_only.tar.gz"
PY_SHA=5d546fc004fc2435fa142eb8f42ed8de5eb4725963dc22f7e6671014a80eb560

[ -x "$HV/bin/harbor" ] || { /usr/bin/python3.12 -m venv "$HV"; "$HV/bin/pip" install -q harbor==0.24.0; }

if [ ! -x "$RT/bin/python3" ]; then
  mkdir -p "$RT/tools" "$RT/bin" "$BASE/dl"
  curl -sL -o "$BASE/dl/tmux.gz" "$TMUX_URL"
  curl -sL -o "$BASE/dl/py.tgz" "$PY_URL"
  echo "$TMUX_SHA  $BASE/dl/tmux.gz" | sha256sum -c -
  echo "$PY_SHA  $BASE/dl/py.tgz" | sha256sum -c -
  gunzip -c "$BASE/dl/tmux.gz" > "$RT/tools/tmux" && chmod +x "$RT/tools/tmux"
  tar xzf "$BASE/dl/py.tgz" -C "$RT" && mv "$RT/python" "$RT/py"
  "$RT/py/bin/python3" -m pip install -q --no-warn-script-location uvicorn fastapi
  ln -sf ../py/bin/python3 "$RT/bin/python3"
fi

# musl build for Alpine images; server deps installed from musllinux wheels by the host pip.
PYM_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.12.15%2B20261013-x86_64-unknown-linux-musl-install_only.tar.gz"
if [ ! -x "$RT/musl/bin/python3" ]; then
  curl -sL -o "$BASE/dl/pym.tgz" "$PYM_URL"
  sha256sum "$BASE/dl/pym.tgz" | tee -a "$RT/SHA256SUMS"
  mkdir -p "$BASE/dl/pym" && tar xzf "$BASE/dl/pym.tgz" -C "$BASE/dl/pym" && mv "$BASE/dl/pym/python" "$RT/musl"
  "$RT/py/bin/python3" -m pip install -q --no-warn-script-location --target "$RT/musl/lib/python3.12/site-packages" \
    --platform musllinux_1_2_x86_64 --python-version 3.12 --only-binary=:all: uvicorn fastapi
fi

# Docker injects these localhost entries into every container's /etc/hosts; apptainer
# --containall does not, which breaks tools that bind or resolve "localhost".
printf '127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n' > "$RT/hosts"

SD=$("$HV/bin/python" -c "import harbor.environments.singularity as s, os; print(os.path.dirname(s.__file__))")
[ -f "$SD/bootstrap.stock.sh" ] || cp "$SD/bootstrap.sh" "$SD/bootstrap.stock.sh"
cp "$(dirname "$0")/../patches/harbor_singularity_bootstrap.sh" "$SD/bootstrap.sh"
chmod +x "$SD/bootstrap.sh" "$SD/bootstrap.stock.sh"
# Writable per-trial sandbox copies + stage the stock bootstrap (see the patch docstring).
"$HV/bin/python" "$(dirname "$0")/../patches/patch_harbor_singularity.py" "$SD/singularity.py"
echo "harbor runtime ready: $RT (mount with --mounts-json '[{\"type\":\"bind\",\"source\":\"$RT\",\"target\":\"/opt/harbor-rt\",\"read_only\":true}]')"
