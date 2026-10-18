#!/bin/bash
# Merge ablation on independent dev sets: build every configuration in
# configs/merge_ablation.tsv, score its per-domain dev loss on one A100, keep recipe + stats,
# delete the weights. 3 GPU lanes on one gpu-a100-dev node. Allocation: the project allocation.
#   sbatch scripts/hpc_merge_ablation.slurm            (from $SCRATCH/qwen36/repo)
#SBATCH -J qwen36-merge-ablation
#SBATCH -p gpu-a100-dev
#SBATCH -N 1 -n 1
#SBATCH -t 02:00:00
#SBATCH -A the project allocation
#SBATCH -o logs/slurm-%x-%j.out
set -u
cd "$SCRATCH/qwen36/repo"
# Lustre ($SCRATCH) random reads make safetensors loading ~17 min per model; stage the base on the
# node-local SSD and write merged checkpoints there.
mkdir -p /tmp/qwen36/hf-cache/hub
cp -rL "$SCRATCH/hf-cache/hub/models--Qwen--Qwen3.5-9B" /tmp/qwen36/hf-cache/hub/
export HF_HOME=/tmp/qwen36/hf-cache HF_HUB_OFFLINE=1 PYTHONPATH=src
PY=$SCRATCH/qwen36/venv/bin/python
OUT=/tmp/qwen36/merged
mkdir -p logs results/dev "$OUT"
lane() {  # lane <gpu> : configs whose row index % 3 == gpu
  local gpu=$1 i=0
  tail -n +2 configs/merge_ablation.tsv | while IFS=$'\t' read -r name args; do
    if [ $((i % 3)) -eq "$gpu" ] && [ ! -f "results/dev/$name.json" ]; then
      model=Qwen/Qwen3.5-9B
      if [ "$args" != "-" ]; then
        model=$OUT/$name
        # shellcheck disable=SC2086
        "$PY" -m takehome.merge $args --out "$model" >> "logs/merge_$name.log" 2>&1
        cp "$model/merge_recipe.json" "results/dev/$name.recipe.json"
        [ -f "$model/merge_stats.json" ] && cp "$model/merge_stats.json" "results/dev/$name.merge_stats.json"
      fi
      rev=(); [ "$model" = Qwen/Qwen3.5-9B ] && rev=(--revision c202236235762e1c871ad0ccb60c8ee5ba337b9a)
      CUDA_VISIBLE_DEVICES=$gpu "$PY" -m takehome.dev_loss --model "$model" "${rev[@]}" --name "$name" \
        --out "results/dev/$name.json" >> "logs/devloss_$name.log" 2>&1
      [ "$model" != Qwen/Qwen3.5-9B ] && rm -r "$model"
    fi
    i=$((i + 1))
  done
}
lane 0 & lane 1 & lane 2 &
wait
