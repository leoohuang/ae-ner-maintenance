#!/bin/bash
#SBATCH -A naiss2026-4-1472-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 05:00:00

set -e
cd ~/ae-ner-maintenance
module purge
ml GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 OpenMPI/5.0.8 PyTorch/2.9.1-CUDA-12.9.1
source ~/venv-gpu/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.13/site-packages:$PWD:$PYTHONPATH
mkdir -p results/gpu

case "$1" in
  random)      python scripts/run_random.py --out results/gpu/random.jsonl ;;
  uncertainty) python scripts/run_active.py --strategy uncertainty --out results/gpu/uncertainty.jsonl ;;
  hybrid)      python scripts/run_active.py --strategy hybrid --out results/gpu/hybrid.jsonl ;;
  *) echo "unknown strategy: $1"; exit 1 ;;
esac