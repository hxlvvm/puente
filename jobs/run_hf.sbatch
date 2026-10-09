#!/bin/bash
#SBATCH --job-name=puente_hf
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=runs/logs/%x_%j.log
# One benchmark configuration with a local open model on one GPU (40 GB is enough for a 7B model in bf16).
# Submit from the repo root, with the model already in the Hugging Face cache:
#   sbatch jobs/run_hf.sbatch --pipeline direct --guard
set -euo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
.venv/bin/python -u -m bench.run --backend hf --model Qwen/Qwen2.5-7B-Instruct --chunk 32 "$@"
