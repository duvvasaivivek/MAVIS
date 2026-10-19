#!/bin/bash
#SBATCH --job-name=mavis-m3-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/lakshum/Duvva_Sai_Vivek/logs/mavis-m3-ablation-%j.out
#SBATCH --error=/home/lakshum/Duvva_Sai_Vivek/logs/mavis-m3-ablation-%j.err

set -e

PROJECT=/home/lakshum/Duvva_Sai_Vivek/MAVIS
VENV=/home/lakshum/Duvva_Sai_Vivek/environments/MAVISvenv

cd "$PROJECT"

source "$VENV/bin/activate"

COMPILER=/home/lakshum/Duvva_Sai_Vivek/environments/compiler
export PATH="$VENV/bin:$COMPILER/bin:$PATH"
export CC=/home/lakshum/Duvva_Sai_Vivek/tools/triton-cc
export CXX="$COMPILER/bin/x86_64-conda-linux-gnu-g++"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "MAVIS Phase 5 M3 Ablation Training (Memory + Attention ONLY, NO EWC)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "=========================================="

nvidia-smi || echo "WARNING: nvidia-smi failed; continuing to CUDA training checks."

echo "=========================================="
echo "Starting MAVIS M3 Ablation (Memory + Attention, NO EWC: --no-ewc --fresh)..."
echo "=========================================="

python main.py train_mavis --no-ewc --fresh

echo "=========================================="
echo "MAVIS M3 Ablation completed"
echo "Finished: $(date)"
echo "=========================================="
