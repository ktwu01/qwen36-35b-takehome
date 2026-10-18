#!/bin/bash
# Serve one evaluated model with vLLM (OpenAI-compatible API). Identical flags for Base,
# experts and merged models; only MODEL/NAME change.
#   bash scripts/serve.sh <model path or HF id> <served name> [port]
# Quantization: online FP8 weights (vLLM --quantization fp8) for every model, so a 9B model
# fits next to a 256K-token KV cache on a 20-24 GB GPU. Thinking mode is on (model default).
set -euo pipefail
MODEL=$1; NAME=$2; PORT=${3:-8000}
VLLM=${VLLM:-/home/users3/kw35262/projects/paul-graham-llm/.venv/bin/vllm}
export HF_HOME=${HF_HOME:-/var/tmp/$USER/hf-cache} HF_HUB_OFFLINE=1
REV_ARGS=()
[[ "$MODEL" == Qwen/* ]] && REV_ARGS=(--revision "${BASE_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}")
exec "$VLLM" serve "$MODEL" "${REV_ARGS[@]}" --served-model-name "$NAME" --port "$PORT" --host 0.0.0.0 \
  --quantization fp8 --dtype bfloat16 --max-model-len 262144 --gpu-memory-utilization 0.92 \
  --max-num-seqs "${MAX_SEQS:-32}" --enable-prefix-caching --seed 0 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image": 1}'
