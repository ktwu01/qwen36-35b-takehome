#!/bin/bash
# Control: one LoRA on the union of the three expert datasets (same LoRA config and data, one
# adapter), DDP on the 3 A100s of a gpu-a100-dev node: per-device batch 1 x grad-accum 5 x 3 GPUs
# = global batch 15 (experts: 16). 2 h jobs, chained; resumes from the last checkpoint.
# NOT what produced the reported union adapter: this DDP run failed at the first step
# (DistributedDataParallel hides get_base_model). The reported run is single-GPU, batch 1 x 16:
#   PYTHONPATH=src python -m takehome.sft --expert union --data data/processed/union --out outputs/union
# (gpu-a100-small VMs mask x86-64-v2 CPU flags, which numpy 2.x requires.) Allocation: the project allocation.
#   sbatch scripts/hpc_train_union.slurm            (from $SCRATCH/qwen36/repo)
#SBATCH -J qwen36-union
#SBATCH -p gpu-a100-dev
#SBATCH -N 1 -n 1
#SBATCH -t 02:00:00
#SBATCH -A the project allocation
#SBATCH -o logs/slurm-%x-%j.out
set -u
cd "$SCRATCH/qwen36/repo"
export HF_HOME=$SCRATCH/hf-cache HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs outputs
[ -f outputs/union/adapter/adapter_model.safetensors ] && { echo "union already done"; exit 0; }
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw --format=csv -l 30 \
  > "logs/gpu_${SLURM_JOB_ID}.csv" &
"$SCRATCH/qwen36/venv/bin/torchrun" --standalone --nproc_per_node 3 -m takehome.sft --expert union \
  --data data/processed/union --out outputs/union --grad-accum 5 >> logs/train_union.log 2>&1
