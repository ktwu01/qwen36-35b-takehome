#!/bin/bash
# Control: one LoRA on the union of the three expert datasets (same config, same data, one
# adapter) on an HPC supercomputer gpu-a100-small (1x A100 40GB). Resumes from the last checkpoint.
# Allocation: the project allocation (user cap 245 SU).
#   sbatch scripts/hpc_train_union.slurm            (from $SCRATCH/qwen36/repo)
#SBATCH -J qwen36-union
#SBATCH -p gpu-a100-small
#SBATCH -N 1 -n 1
#SBATCH -t 20:00:00
#SBATCH -A the project allocation
#SBATCH -o logs/slurm-%x-%j.out
set -u
cd "$SCRATCH/qwen36/repo"
export HF_HOME=$SCRATCH/hf-cache HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs outputs
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw --format=csv -l 30 \
  > "logs/gpu_${SLURM_JOB_ID}.csv" &
"$SCRATCH/qwen36/venv/bin/python" -m takehome.sft --expert union --data data/processed/union --out outputs/union \
  >> logs/train_union.log 2>&1
