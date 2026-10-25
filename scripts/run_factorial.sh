#!/bin/bash
# =============================================================================
# MAVIS factorial study — single SLURM job, all cells trained sequentially
# =============================================================================
# One job, one GPU, same shape as scripts/run_phase5.sh. It walks the five
# cells of the factorial for each seed in turn:
#
#   M1     baseline CNN                        -> baseline{tag}_results.json
#   M2     CNN + EWC                           -> ewc{tag}_results.json
#   M3     memory + attention, no EWC          -> mavis_m3{tag}_results.json
#   M4     full MAVIS                          -> mavis{tag}_results.json
#   M4noA  full MAVIS, retrieval path bypassed -> mavis_noattn{tag}_results.json
#
# SUBMIT
#   # preflight on a login node first (2 minutes, no GPU needed):
#   python tests/test_review_sanity.py && python tests/test_decode_cache.py
#
#   sbatch scripts/run_factorial.sh                      # published scale, all 5 cells x 3 seeds
#   SCALE=quick sbatch scripts/run_factorial.sh          # width 32 / 5 tasks / 10 epochs
#   SEEDS="1" sbatch scripts/run_factorial.sh            # one seed only
#   CELLS="M3 M4 M4noA" sbatch scripts/run_factorial.sh  # just the MAVIS cells
#   EWC_LAMBDA=50000 TAG_EXTRA=_lam50k CELLS="M2" sbatch scripts/run_factorial.sh
#
# IF THE WALL CLOCK CUTS IT SHORT
#   Just resubmit. Any cell whose result file already exists is skipped, so a
#   second submission picks up where the first stopped. Nothing is recomputed
#   and nothing is overwritten. If you would rather not rely on that, split the
#   work across submissions with SEEDS= or CELLS= as shown above.
#
# ANALYSE
#   python experiments/analyze_factorial.py --width 128 --tasks 10 --epochs 50
#   python experiments/analyze_factorial.py --width 32 --tasks 5 --epochs 10   # quick
#   (the job runs this itself at the end, so it is already in the .out file)
#
# NOTES
#   * One cell failing does not abort the rest. Failures are collected and
#     listed in a summary at the end, and the job exits non-zero if any failed.
#   * --fresh is deliberately NOT passed, so a cell interrupted mid-way resumes
#     from its highest completed task on the next submission. To force a real
#     restart, delete that cell's result file and its checkpoint directory.
#   * Checkpoints written before the 2026-10-25 fixes must not be resumed onto.
#     Use a tag you have not used before, or clear the directory first.
#   * The first cell pays the image decode cost, about 235 s per task split.
#     Every later cell reads the cache in about 0.1 s, so cell one looks
#     disproportionately slow and the rest will not.
# =============================================================================

#SBATCH --job-name=mavis-factorial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/lakshum/Duvva_Sai_Vivek/logs/mavis-factorial-%j.out
#SBATCH --error=/home/lakshum/Duvva_Sai_Vivek/logs/mavis-factorial-%j.err

PROJECT=/home/lakshum/Duvva_Sai_Vivek/MAVIS
VENV=/home/lakshum/Duvva_Sai_Vivek/environments/MAVISvenv

cd "$PROJECT" || exit 1

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

SEEDS="${SEEDS:-1 2 3}"
CELLS="${CELLS:-M1 M2 M3 M4 M4noA}"
EWC_LAMBDA="${EWC_LAMBDA:-5000}"
TAG_EXTRA="${TAG_EXTRA:-}"

echo "=========================================="
echo "MAVIS factorial study"
echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Python:     $(python --version 2>&1)"
echo "PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "GPU:        $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "CUDA:       $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Devices:    ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-unset}"
echo "------------------------------------------"
echo "Scale:      width $WIDTH | $TASKS tasks | $EPOCHS epochs/task | batch $BATCH"
echo "Cells:      $CELLS"
echo "Seeds:      $SEEDS"
echo "EWC lambda: $EWC_LAMBDA   (the published M2 used 50000; 5000 matches M4)"
echo "=========================================="

nvidia-smi || echo "WARNING: nvidia-smi failed; continuing to the CUDA checks above."

FAILED=""
DONE_COUNT=0
SKIP_COUNT=0
RUN_START=$(date +%s)

for SEED in $SEEDS; do
    for CELL in $CELLS; do

        case "$CELL" in
            M1)    COMMAND="train_baseline"; PREFIX="baseline";     EXTRA="" ;;
            M2)    COMMAND="train_ewc";      PREFIX="ewc";          EXTRA="--ewc-lambda $EWC_LAMBDA" ;;
            M3)    COMMAND="train_mavis";    PREFIX="mavis_m3";     EXTRA="--no-ewc" ;;
            M4)    COMMAND="train_mavis";    PREFIX="mavis";        EXTRA="" ;;
            M4noA) COMMAND="train_mavis";    PREFIX="mavis_noattn"; EXTRA="--no-attention" ;;
            *)     echo "[error] unknown cell '$CELL' — skipping"; FAILED="$FAILED ${CELL}/s${SEED}"; continue ;;
        esac

        TAG="_fac${TAG_EXTRA}_w${WIDTH}_t${TASKS}_e${EPOCHS}_s${SEED}"
        RESULT="results/raw/${PREFIX}${TAG}_results.json"

        echo ""
        echo "=========================================="
        echo ">>> $CELL   seed $SEED   ($(date +%H:%M:%S))"
        echo "    tag    $TAG"
        echo "    result $RESULT"
        echo "=========================================="

        if [ -f "$RESULT" ]; then
            echo "[skip] already exists — delete it and its checkpoint directory to re-run"
            SKIP_COUNT=$(( SKIP_COUNT + 1 ))
            continue
        fi

        CELL_START=$(date +%s)
        # shellcheck disable=SC2086
        python main.py "$COMMAND" \
            --seed "$SEED" \
            --tasks "$TASKS" \
            --epochs "$EPOCHS" \
            --width "$WIDTH" \
            --batch-size "$BATCH" \
            --tag "$TAG" \
            $EXTRA
        STATUS=$?
        CELL_MINS=$(( ( $(date +%s) - CELL_START ) / 60 ))

        if [ "$STATUS" -eq 0 ]; then
            echo "[done] $CELL seed $SEED in ${CELL_MINS} min -> $RESULT"
            DONE_COUNT=$(( DONE_COUNT + 1 ))
        else
            echo "[FAIL] $CELL seed $SEED exited $STATUS after ${CELL_MINS} min; continuing with the rest"
            FAILED="$FAILED ${CELL}/s${SEED}"
        fi
    done
done

TOTAL_MINS=$(( ( $(date +%s) - RUN_START ) / 60 ))

echo ""
echo "=========================================="
echo "Factorial study finished: $(date)"
echo "Total wall time: ${TOTAL_MINS} min"
echo "Trained: $DONE_COUNT   Skipped (already present): $SKIP_COUNT"
if [ -n "$FAILED" ]; then
    echo "FAILED cells:$FAILED"
else
    echo "No failures."
fi
echo "=========================================="
echo ""
echo "Analysis:"
python experiments/analyze_factorial.py --width "$WIDTH" --tasks "$TASKS" --epochs "$EPOCHS" \
    || echo "(analysis step failed; run it by hand)"
echo "=========================================="

if [ -n "$FAILED" ]; then
    exit 1
fi
