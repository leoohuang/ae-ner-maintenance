#!/bin/bash
#SBATCH -A naiss2026-4-1472-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 05:00:00
# Human-label runs: sbatch scripts/gpu_job.sh random|uncertainty|hybrid
# LLM-label runs:   sbatch scripts/gpu_job.sh rq2 | rq3 <model> <strategy> | rq3b <model>

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
  # LLM annotation protocol (README amendments 2 and 8)
  rq2)   for m in gpt-6-astra gpt-6-luna claude-opus-5-5 claude-haiku-5-5 gemini-4-carbon qwen4-max qwen3.8-27b deepseek-v4-pro; do
           python scripts/rq2_uncertainty.py --model $m
         done ;;
  rq3)   # sbatch scripts/gpu_job.sh rq3 <model> random|uncertainty|hybrid
         mkdir -p results/gpu/llm/$2
         if [ "$3" = random ]; then
           python scripts/run_random.py --labels data/llm/$2.jsonl --out results/gpu/llm/$2/random.jsonl
         else
           python scripts/run_active.py --strategy $3 --labels data/llm/$2.jsonl --out results/gpu/llm/$2/$3.jsonl
         fi ;;
  rq3b)  python scripts/run_review.py --model $2 ;;
  *) echo "unknown job: $1"; exit 1 ;;
esac