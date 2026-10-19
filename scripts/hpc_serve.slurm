#!/bin/bash
# Eval serving on one an HPC supercomputer gpu-a100 node (3x A100 40GB): one vLLM server per GPU with the
# same flags as scripts/serve.sh, each exposed on seismic18 at 127.0.0.1:1810<k+1> through a
# restricted reverse SSH tunnel (key may only open those listeners). K0 (first slot, default 0) and
# NGPU (default 3) via sbatch --export=ALL,K0=6,NGPU=1 select slots k = K0 + gpu, so several nodes
# (gpu-a100 or 1-GPU gpu-a100-small) can serve at once. A supervisor loop
# serves whatever $SCRATCH/qwen36/serve/gpu<k>.model names ("<served name> <path or HF id>"), so GPUs
# can be retargeted (base -> expert -> merged) by editing that file. Allocation: the project allocation.
#   sbatch scripts/hpc_serve.slurm            (from $SCRATCH/qwen36/repo)
#SBATCH -J qwen36-serve
#SBATCH -p gpu-a100
#SBATCH -N 1 -n 1
#SBATCH -t 24:00:00
#SBATCH -A the project allocation
#SBATCH -o logs/slurm-%x-%j.out
set -u
cd "$SCRATCH/qwen36/repo"
K0=${K0:-0}; NGPU=${NGPU:-3}
V=$SCRATCH/qwen36/vserve
CTRL=$SCRATCH/qwen36/serve
mkdir -p "$CTRL" logs
# Stage weights on the node-local SSD: safetensors loading from Lustre takes ~17 min per model.
mkdir -p /tmp/qwen36/hf-cache/hub /tmp/qwen36/models
cp -rL "$SCRATCH/hf-cache/hub/models--Qwen--Qwen3.5-9B" /tmp/qwen36/hf-cache/hub/
export HF_HOME=/tmp/qwen36/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=$SCRATCH/qwen36/cuda-home PATH=$V/bin:$SCRATCH/qwen36/cuda-home/bin:$PATH
export LIBRARY_PATH=$SCRATCH/qwen36/cuda-home/lib64
export LD_LIBRARY_PATH=/opt/apps/gcc/9.4.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}  # numpy built from source with gcc 9.4
# Self-resubmitting dev windows (so the chain needs no login session): if $CTRL/RESUBMIT_DEV exists
# (it holds a cutoff time), this is a gpu-a100-dev job, it is before the cutoff, the window budget in
# $CTRL/RESUBMIT_DEV_LEFT (windows, 6 SU each; set from the login node, where the the project allocation balance is
# visible: the allocation balance command does not exist on compute nodes) is > 0, and no other serve job is pending there,
# queue the next window now (afterany this one) and decrement the budget.
left=$(cat "$CTRL/RESUBMIT_DEV_LEFT" 2>/dev/null || echo 0)
if [ -f "$CTRL/RESUBMIT_DEV" ] && [ "$SLURM_JOB_PARTITION" = gpu-a100-dev ] \
   && [ "$(date +%s)" -lt "$(date -d "$(cat "$CTRL/RESUBMIT_DEV")" +%s)" ] && [ "$left" -gt 0 ] \
   && [ -z "$(squeue -u "$USER" -h -t PD -n qwen36-serve -p gpu-a100-dev -o %i)" ]; then
  sbatch -p gpu-a100-dev -t 02:00:00 -d "afterany:$SLURM_JOB_ID" --export=ALL,K0=$K0,NGPU=$NGPU \
    scripts/hpc_serve.slurm >> "logs/resubmit_${SLURM_JOB_ID}.log" 2>&1 \
    && echo $((left - 1)) > "$CTRL/RESUBMIT_DEV_LEFT"
fi
HF_HOME=/tmp/qwen36/hf-cache bash scripts/hpc_build_models.sh >> "logs/build_models_${SLURM_JOB_ID}.log" 2>&1
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw --format=csv -l 60 \
  > "logs/gpu_${SLURM_JOB_ID}.csv" &

serve_gpu() {  # serve_gpu <i>: restart vLLM on GPU i whenever its control file changes
  local i=$1 k=$((K0 + $1)) port=$((18000 + $1)) cur="" pid=""
  while true; do
    want=$(cat "$CTRL/gpu$k.model" 2>/dev/null || true)
    # relaunch on a control-file change, or when the server died (crash, venv not ready yet)
    if [ "$want" != "$cur" ] || { [ -n "$want" ] && ! kill -0 "${pid:-0}" 2>/dev/null; }; then
      [ -n "$pid" ] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null
      pid=""; cur=$want
      if [ -n "$want" ]; then
        read -r name model <<< "$want"
        if [[ "$model" == /* ]] && [ ! -d "$model" ]; then  # e.g. merged chosen after the job started
          flock /tmp/qwen36/build.lock env HF_HOME=/tmp/qwen36/hf-cache bash scripts/hpc_build_models.sh \
            >> "logs/build_models_${SLURM_JOB_ID}.log" 2>&1
        fi
        if [ -d "$model" ]; then  # a checkpoint directory: copy it to local SSD once
          [ -d "/tmp/qwen36/models/$name" ] || cp -r "$model" "/tmp/qwen36/models/$name"
          model=/tmp/qwen36/models/$name
        fi
        rev=(); [[ "$model" == Qwen/* ]] && rev=(--revision c202236235762e1c871ad0ccb60c8ee5ba337b9a)
        CUDA_VISIBLE_DEVICES=$i "$V/bin/vllm" serve "$model" "${rev[@]}" --served-model-name "$name" \
          --port "$port" --host 127.0.0.1 --quantization fp8 --kv-cache-dtype fp8 --dtype bfloat16 \
          --max-model-len 262144 --gpu-memory-utilization 0.92 --max-num-seqs 32 --enable-prefix-caching \
          --seed 0 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
          --limit-mm-per-prompt '{"image": 1}' >> "logs/vllm_hpc_gpu${k}_${name}.log" 2>&1 &
        pid=$!
      fi
    fi
    sleep 30
  done
}

tunnel() {  # tunnel <i>: keep 127.0.0.1:1810<k+1> on seismic18 -> local vLLM port of GPU i
  local i=$1 k=$((K0 + $1))
  while true; do
    ssh -i ~/.ssh/id_ed25519_seismic_tunnel -o BatchMode=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N \
      -R "127.0.0.1:$((18101 + k)):127.0.0.1:$((18000 + i))" kw35262@seismic18.geo.utexas.edu \
      >> "logs/tunnel_${SLURM_JOB_ID}_$i.log" 2>&1
    sleep 10
  done
}

for i in $(seq 0 $((NGPU - 1))); do serve_gpu "$i" & tunnel "$i" & done
wait
