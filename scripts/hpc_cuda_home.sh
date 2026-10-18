#!/bin/bash
# Assemble a CUDA 12.9 toolchain dir (CUDA_HOME) from the pip CUDA packages in the HPC serving
# venv, for FlashInfer's JIT (FP8 KV-cache attention). HPC has no CUDA 12.9 toolkit module.
#   bash scripts/hpc_cuda_home.sh $SCRATCH/qwen36/vserve $SCRATCH/qwen36/cuda-home
set -euo pipefail
V=$1; H=$2
N=$V/lib/python3.12/site-packages/nvidia
mkdir -p "$H/include" "$H/lib64"
ln -sfn "$N/cuda_nvcc/bin" "$H/bin"
ln -sfn "$N/cuda_nvcc/nvvm" "$H/nvvm"
for d in cuda_nvcc cuda_runtime cuda_cccl; do
  for f in "$N/$d/include/"*; do [ -e "$H/include/$(basename "$f")" ] || ln -s "$f" "$H/include/"; done
done
for f in "$N/cuda_runtime/lib/"*; do ln -sfn "$f" "$H/lib64/"; done
ln -sfn "$N/cuda_runtime/lib/libcudart.so.12" "$H/lib64/libcudart.so"
"$H/bin/nvcc" --version | tail -2
