#!/bin/bash
# Build the full-weight checkpoints that HPC serving needs, on the compute node (fast local CPU),
# into $SCRATCH/qwen36/models/{expert-swe,expert-term,expert-sci,merged}. Skips existing ones.
# Deterministic: same adapters + configs/merge_final.json + seed give the same weights as seismic.
set -u
cd "$SCRATCH/qwen36/repo"
export PYTHONPATH=src
PY=$SCRATCH/qwen36/venv/bin/python
M=$SCRATCH/qwen36/models
mkdir -p "$M" /tmp/qwen36/build
for e in swe term sci; do
  if [ ! -f "$M/expert-$e/merge_recipe.json" ] && [ -f "outputs/$e/adapter/adapter_model.safetensors" ]; then
    "$PY" -m takehome.merge --method single --experts "outputs/$e" --out "/tmp/qwen36/build/expert-$e" \
      && cp -r "/tmp/qwen36/build/expert-$e" "$M/"
  fi
done
if [ -f configs/merge_final.json ] && [ ! -f "$M/merged/merge_recipe.json" ]; then
  args=$("$PY" -c "import json;print(json.load(open('configs/merge_final.json'))['merge_args'])")
  # shellcheck disable=SC2086
  "$PY" -m takehome.merge $args --out /tmp/qwen36/build/merged && cp -r /tmp/qwen36/build/merged "$M/"
fi
ls "$M"
