#!/bin/bash
# =============================================================================
# MAVIS factorial study — SLURM job array
# =============================================================================
# Runs all five cells of the 2x2 factorial (plus the attention ablation) across
# several seeds, one array task per (cell, seed), after the 2026-10-25 fixes.
#
#   M1     baseline CNN                        -> baseline{tag}_results.json
#   M2     CNN + EWC                           -> ewc{tag}_results.json
#   M3     memory + attention, no EWC          -> mavis_m3{tag}_results.json
#   M4     full MAVIS                          -> mavis{tag}_results.json
#   M4noA  full MAVIS, retrieval path bypassed -> mavis_noattn{tag}_results.json
#
# SUBMIT
#   # 1. preflight on a login node or a short interactive allocation (2 min, no GPU needed):
#   python tests/test_review_sanity.py && python tests/test_decode_cache.py
#
#   # 2. cheap sweep first — 5 cells x 3 seeds at reduced scale, ~25 min per cell.
#   #    This is what produces the seed-to-seed spread. Do this before spending
#   #    the full-scale GPU hours.
#   SCALE=quick sbatch scripts/run_factorial.sh
#
#   # 3. full published scale — width 128, 10 tasks, 50 epochs, ~4 h per MAVIS cell.
#   sbatch scripts/run_factorial.sh
#
#   # single cell / single seed, e.g. just M4 seed 1:
#   sbatch --array=3 scripts/run_factorial.sh
#
#   # the EWC strength variant (published M2 used 50000; the default here matches M4):
#   EWC_LAMBDA=50000 TAG_EXTRA=_lam50k sbatch --array=1,6,11 scripts/run_factorial.sh
#
# ANALYSE (after the array finishes)
#   python experiments/analyze_factorial.py --width 128 --tasks 10 --epochs 50
#   python experiments/analyze_factorial.py --width 32 --tasks 5 --epochs 10   # quick sweep
#
# NOTES
#   * Each array task is an independent process with its own tagged checkpoint
#     directory, so they are safe to run concurrently on separate GPUs.
#   * The decoded-image disk cache is shared. It is written to a pid-suffixed
#     temporary file and then renamed atomically, so concurrent array tasks
#     cannot corrupt it; the first one to reach a task pays the decode cost and
#     the rest load it in about 0.1 s.
#   * --fresh is deliberately NOT passed, so a requeued or extended job resumes
#     from its highest completed task. A cell whose result file already exists is
#     skipped entirely. If you want a genuine restart, delete that cell's result
#     file and its checkpoint directory first.
#   * Checkpoints written before the 2026-10-25 fixes must not be resumed onto.
#     Use a tag you have not used before, or clear the directory.
# =============================================================================

#SBATCH --job-name=mavis-factorial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/home/lakshum/Duvva_Sai_Vivek/logs/mavis-factorial-%A_%a.out
#SBATCH --error=/home/lakshum/Duvva_Sai_Vivek/logs/mavis-factorial-%A_%a.err

set -euo pipefail

PROJECT=/home/lakshum/Duvva_Sai_Vivek/MAVIS
VENV=/home/lakshum/Duvva_Sai_Vivek/environments/MAVISvenv

cd "$PROJECT"
source "$VENV/bin/activate"

COMPILER=/home/lakshum/Duvva_Sai_Vivek/environments/compiler
export PATH="$VENV/bin:$COMPILER/bin:$PATH"
export CC=/home/lakshum/Duvva_Sai_Vivek/tools/triton-cc
export CXX="$COMPILER/bin/x86_64-conda-linux-gnu-g++"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ----------------------------------------------------------------------------
# Scale. SCALE=quick runs the reduced sweep; anything else runs published scale.
# ----------------------------------------------------------------------------
SCALE="${SCALE:-full}"
if [ "$SCALE" = "quick" ]; then
    WIDTH=32; TASKS=5; EPOCHS=10; BATCH=128
else
    WIDTH=128; TASKS=10; EPOCHS=50; BATCH=128
fi

SEEDS=(1 2 3)
CELLS=(M1 M2 M3 M4 M4noA)

EWC_LAMBDA="${EWC_LAMBDA:-5000}"
TAG_EXTRA="${TAG_EXTRA:-}"

# ----------------------------------------------------------------------------
# Map the array index onto one (cell, seed) pair.
# ----------------------------------------------------------------------------
IDX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_CELLS=${#CELLS[@]}
CELL="${CELLS[$(( IDX % NUM_CELLS ))]}"
SEED="${SEEDS[$(( IDX / NUM_CELLS ))]}"

TAG="_fac${TAG_EXTRA}_w${WIDTH}_t${TASKS}_e${EPOCHS}_s${SEED}"

case "$CELL" in
    M1)    COMMAND="train_baseline"; EXTRA=(); PREFIX="baseline" ;;
    M2)    COMMAND="train_ewc";      EXTRA=(--ewc-lambda "$EWC_LAMBDA"); PREFIX="ewc" ;;
    M3)    COMMAND="train_mavis";    EXTRA=(--no-ewc); PREFIX="mavis_m3" ;;
    M4)    COMMAND="train_mavis";    EXTRA=(); PREFIX="mavis" ;;
    M4noA) COMMAND="train_mavis";    EXTRA=(--no-attention); PREFIX="mavis_noattn" ;;
    *)     echo "Unknown cell '$CELL' from array index $IDX"; exit 1 ;;
esac

RESULT="results/raw/${PREFIX}${TAG}_results.json"

echo "=========================================="
echo "MAVIS factorial study"
echo "=========================================="
echo "Job:        ${SLURM_ARRAY_JOB_ID:-local}_${IDX}"
echo "Cell:       $CELL   (seed $SEED)"
echo "Scale:      width $WIDTH | $TASKS tasks | $EPOCHS epochs/task | batch $BATCH"
echo "Tag:        $TAG"
echo "Result:     $RESULT"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Python:     $(python --version 2>&1)"
echo "PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "GPU:        $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "CUDA:       $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Devices:    ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-unset}"
if [ "$CELL" = "M2" ]; then
    echo "EWC lambda: $EWC_LAMBDA   (the published M2 used 50000; 5000 matches M4)"
fi
echo "=========================================="

nvidia-smi || echo "WARNING: nvidia-smi failed; continuing to the CUDA checks above."

if [ -f "$RESULT" ]; then
    echo "[skip] $RESULT already exists. Delete it and the matching checkpoint"
    echo "       directory if you want this cell re-run."
    exit 0
fi

echo "Starting $CELL, seed $SEED ..."
echo "=========================================="

python main.py "$COMMAND" \
    --seed "$SEED" \
    --tasks "$TASKS" \
    --epochs "$EPOCHS" \
    --width "$WIDTH" \
    --batch-size "$BATCH" \
    --tag "$TAG" \
    ${EXTRA[@]+"${EXTRA[@]}"}
    # the ${...+...} form is needed because `set -u` on bash 4.2 (CentOS 7 and
    # similar) treats an empty array expansion as an unbound variable

echo "=========================================="
echo "$CELL seed $SEED completed"
echo "Wrote: $RESULT"
echo "Finished: $(date)"
echo "=========================================="
