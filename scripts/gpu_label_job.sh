#!/bin/bash
# LLM annotation protocol: Qwen/Qwen3.8-27B (BF16, self-hosted) labels with the frozen prompt v2.
# Download the weights once on the login node first (compute nodes may have no internet):
#   bash scripts/gpu_label_job.sh download
# Then:
#   sbatch scripts/gpu_label_job.sh tune         # 20 tuning records: check that it works
#   sbatch scripts/gpu_label_job.sh train_pool   # the 758 train_pool records
#SBATCH -A naiss2026-4-1472-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 04:00:00

set -e
cd ~/ae-ner-maintenance
module purge
ml GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 OpenMPI/5.0.8 PyTorch/2.9.1-CUDA-12.9.1
source ~/venv-gpu/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.13/site-packages:$PWD:$PYTHONPATH
export HF_HOME=${HF_HOME:-/nobackup/proj/disk/naiss2026-4-1472/personal/$USER/hf}   # about 56 GB; home has only 30 GB

case "$1" in
  download)        hf download Qwen/Qwen3.8-27B ;;
  tune|train_pool) export HF_HUB_OFFLINE=1
                   nvidia-smi --query-gpu=name,memory.total --format=csv
                   python scripts/label_llm.py --model qwen3.8-27b --split "$1" ;;
  *) echo "usage: $0 download|tune|train_pool"; exit 1 ;;
esac
