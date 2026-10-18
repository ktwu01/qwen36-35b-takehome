#!/bin/bash
# Assemble a CUDA 12.9 toolchain dir (CUDA_HOME) for FlashInfer's JIT (FP8 KV-cache attention) on
# HPC, which has no CUDA 12.9 toolkit module and whose pip CUDA 12 packages ship no nvcc. Uses
# NVIDIA's component redistributable archives (redistrib_12.9.1.json), checksum-verified.
#   bash scripts/hpc_cuda_home.sh $SCRATCH/qwen36/cuda-home
set -euo pipefail
H=$1
R=https://developer.download.nvidia.com/compute/cuda/redist
mkdir -p "$H" "$H.dl"
while read -r path sha; do
  f="$H.dl/$(basename "$path")"
  [ -f "$f" ] || curl -sL -o "$f" "$R/$path"
  echo "$sha  $f" | sha256sum -c -
  tar xJf "$f" -C "$H.dl"
  d="$H.dl/$(basename "$path" .tar.xz)"
  cp -a "$d/." "$H/"
done <<'EOF'
cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz 7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35
cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz 1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99
cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.9.27-archive.tar.xz 8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa
EOF
[ -d "$H/lib64" ] || ln -s lib "$H/lib64"
"$H/bin/nvcc" --version | tail -2
