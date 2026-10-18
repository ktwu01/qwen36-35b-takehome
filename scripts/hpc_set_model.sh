#!/bin/bash
# Point HPC serving GPU slot k (0-5) at a model: bash scripts/hpc_set_model.sh <k> <served name> <path|HF id>
# The supervisor in hpc_serve.slurm restarts that GPU's vLLM server within 30 s.
set -euo pipefail
k=$1; shift
mkdir -p "$SCRATCH/qwen36/serve"
printf '%s %s\n' "$1" "$2" | tee "$SCRATCH/qwen36/serve/gpu$k.model"
