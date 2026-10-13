#!/bin/bash
#SBATCH --job-name=mavis-phase5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/lakshum/Duvva_Sai_Vivek/logs/mavis-phase5-%j.out
#SBATCH --error=/home/lakshum/Duvva_Sai_Vivek/logs/mavis-phase5-%j.err

set -e

PROJECT=/home/lakshum/Duvva_Sai_Vivek/MAVIS
VENV=/home/lakshum/Duvva_Sai_Vivek/environments/MAVISvenv

cd "$PROJECT"

source "$VENV/bin/activate"

COMPILER=/home/lakshum/Duvva_Sai_Vivek/environments/compiler
export PATH="$VENV/bin:$COMPILER/bin:$PATH"
export CC=/home/lakshum/Duvva_Sai_Vivek/tools/triton-cc
export CXX="$COMPILER/bin/x86_64-conda-linux-gnu-g++"

echo "=========================================="
echo "MAVIS Phase 5 Training (Full MAVIS: Memory + Attention + EWC)"
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
echo "Starting MAVIS Phase 5 (EWC lambda=2000.0, gamma=0.95)..."
echo "=========================================="

python main.py train_mavis --fresh

echo "=========================================="
echo "MAVIS Phase 5 completed"
echo "Finished: $(date)"
echo "=========================================="
