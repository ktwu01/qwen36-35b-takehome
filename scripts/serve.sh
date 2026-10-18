#!/bin/bash
# Serve one evaluated model with vLLM (OpenAI-compatible API). Identical flags for Base,
# experts and merged models; only MODEL/NAME change.
#   bash scripts/serve.sh <model path or HF id> <served name> [port]
# Quantization: online FP8 weights (vLLM --quantization fp8) and FP8 (e4m3) KV cache for every
# model and benchmark, so a 9B model holds a 262,144-token context on a 20-24 GB GPU.
# Thinking mode is on (model default). Run scripts/setup_serving.sh once per host first.
set -euo pipefail
MODEL=$1; NAME=$2; PORT=${3:-18000}
V=${VENV:-/var/tmp/$USER/venvs/train}
export HF_HOME=${HF_HOME:-/var/tmp/$USER/hf-cache} HF_HUB_OFFLINE=1
# No system CUDA toolkit on the hosts: FlashInfer JIT (FP8-KV attention) uses the pip CUDA 13
# toolchain inside the venv; sampling uses vLLM's PyTorch top-k/top-p sampler.
C=$V/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$C PATH=$V/bin:$C/bin:$PATH LIBRARY_PATH=$C/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export VLLM_USE_FLASHINFER_SAMPLER=0
REV_ARGS=()
[[ "$MODEL" == Qwen/* ]] && REV_ARGS=(--revision "${BASE_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}")
exec "$V/bin/vllm" serve "$MODEL" "${REV_ARGS[@]}" --served-model-name "$NAME" --port "$PORT" --host 0.0.0.0 \
  --quantization fp8 --kv-cache-dtype fp8 --dtype bfloat16 --max-model-len 262144 \
  --gpu-memory-utilization 0.92 --max-num-seqs "${MAX_SEQS:-32}" --enable-prefix-caching --seed 0 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image": 1}'
