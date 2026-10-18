#!/bin/bash
# Diagnose the union job's numpy import failure on gpu-a100-small (virtual nodes).
#SBATCH -J probe-small
#SBATCH -p gpu-a100-small
#SBATCH -N 1 -n 1
#SBATCH -t 00:05:00
#SBATCH -A the project allocation
#SBATCH -o logs/probe-small-%j.out
lscpu | grep -E "Model name"
grep -o -E "avx2|avx512f|fma" /proc/cpuinfo | sort | uniq -c
echo "PYTHONPATH=${PYTHONPATH:-}"
env -u PYTHONPATH "$SCRATCH/qwen36/venv/bin/python" -X faulthandler -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | tail -5
"$SCRATCH/qwen36/venv/bin/python" -c "import numpy; print('numpy with PYTHONPATH', numpy.__version__)" 2>&1 | tail -3
