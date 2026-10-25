# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MAVIS (Memory Augmented Visual Incremental System) is a from-scratch PyTorch research codebase for class-incremental continual learning on Tiny ImageNet (200 classes, 64x64, split into 10 sequential tasks of 20 classes, seed 42). No pretrained weights are allowed. The research question: can an external attention-retrieved memory plus EWC parameter regularization reduce catastrophic forgetting under a fixed memory budget?

Four models form a 2x2 factorial: M1 baseline CNN, M2 CNN+EWC, M3 memory+attention only, M4 full MAVIS (memory + attention + EWC). Results and decision history live in `PROJECT_LOG.md` (append there when results change); long-form theory lives in `docs/`. Do not copy result tables into this file.

## Commands

Use the project venv interpreter (`.venv\Scripts\python`, Python 3.10, torch 2.5.1+cu121). There is no pytest, linter, or formatter configured.

First-time setup:

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python main.py setup
```

Entry points:

```bash
python main.py setup                 # verify torch/CUDA, download dataset if missing
python main.py config                # print active config
python main.py visualize             # write results/figures/task_samples.png
python main.py train_baseline        # M1 -> results/raw/baseline_results.json
python main.py train_ewc             # M2 -> results/raw/ewc_results.json
python main.py train_mavis           # M4 -> results/raw/mavis_results.json
python main.py train_mavis --no-ewc  # M3 -> results/raw/mavis_m3_results.json
```

Flags: `--config <yaml>` and `--epochs N` (default 50 per task) apply to every command. `--ewc-lambda F` affects `train_ewc` and `train_mavis`. `--no-ewc` and `--fresh` (delete checkpoints and start at Task 1) only affect `train_mavis`. A full 10-task MAVIS run is ~4 hours on an RTX 6000 Ada. For smoke tests use `--epochs 2`; there is no flag to limit the task count (the `smoke_test_*` fields in `TrainingConfig` are not read anywhere), so a smoke run still iterates all 10 tasks.

Tests are plain scripts, not a pytest suite. Run one directly:

```bash
python tests/test_hierarchical_bayes.py   # guards the Bayes conditional routing identity
python tests/test_gpu_speed.py            # CPU vs GPU matmul sanity check
```

Other tools:
- `python experiments/benchmark.py` prints the M1..M4 markdown comparison table from `results/raw/*.json`.
- `python config.py` regenerates `configs/*.yaml` from the dataclass defaults. `python data/task_generator.py` regenerates `configs/task_definitions.json`.
- `scripts/run_phase5.sh` and `scripts/run_phase5_m3_ablation.sh` are SLURM job scripts with hardcoded cluster paths. They are the HPC entrypoints and will not run locally.
- `scripts/run_factorial.sh` is a single SLURM job, same shape as the Phase 5 scripts, that trains all five cells (M1, M2, M3, M4, M4-no-attention) sequentially for each seed. `sbatch scripts/run_factorial.sh` runs published scale; `SCALE=quick sbatch ...` runs width 32 / 5 tasks / 10 epochs; `SEEDS="1"` and `CELLS="M3 M4"` narrow the work. It skips cells whose result file already exists, so resubmitting after a wall-clock kill continues where it stopped, and one failing cell does not abort the rest. It runs `experiments/analyze_factorial.py` itself at the end, so the summary lands in the job's `.out` file.
- `scripts/generate_chronicle_pdf.py` expects `MAVIS_COMPREHENSIVE_CHRONICLE.md` at the project root, which no longer exists. Treat it as stale.

## Architecture

### Two model/trainer pairs

- `models/baseline.py` `BaselineCNN` + `continual/trainer.py` `ContinualTrainer` serve M1 and M2. EWC is on whenever `ewc_lambda > 0`.
- `models/mavis.py` `MAVIS` + `continual/mavis_trainer.py` `MAVISTrainer` serve M3 and M4. `--no-ewc` sets `config.ewc.enabled = False`, which also selects the checkpoint dir (`results/checkpoints/mavis` vs `mavis_m3`) and the results filename.

Both pairs share `CNNEncoder` (ResNet-18 variant with GroupNorm and Squeeze-Excitation, no stem pooling) and `IncrementalClassifier` (cosine-linear head over all 200 classes).

### Label and task-index conventions

- `TinyImageNetTaskDataset` yields local labels 0..19. Trainers convert to absolute class ids with `task_id * classes_per_task` before computing loss or accuracy.
- `TaskGenerator.get_task()` is 1-indexed; `TinyImageNetLoader` and both trainers are 0-indexed and add 1 internally.
- The classifier always emits 200 logits. Task masking (slicing to active or seen classes) happens in the trainer, never inside the model.

### MAVIS forward contract

`MAVIS.forward(x, task_id=T)` during training retrieves locked prototypes from tasks `< T` plus the EMA running prototypes of task `T`. `forward(x, task_id=None)` at evaluation retrieves from every populated task. `MemoryBank` state is registered buffers, so it is saved and restored through `state_dict`.

### Per-task lifecycle in MAVISTrainer.train_task

1. Train with the combined loss: task-masked CE, feature distillation to both the Task 1 anchor encoder E_0 and the previous encoder E_{t-1}, PODNet spatial distillation, InfoNCE prototype alignment, top-K repulsion from past prototypes, attention self-reconstruction, Gaussian prototype replay on the head, and the Online EWC penalty.
2. Commit K-means prototypes (K=10 per class) and per-class variances for the finished task.
3. Re-encode every past task's prototypes through the current encoder (fixes prototype drift).
4. `align_weights` on the classifier, then balanced classifier finetuning on Gaussian-perturbed prototypes with the encoder, attention, and gate frozen.
5. Snapshot E_0 (after Task 1 only) and E_{t-1}; update the Online EWC Fisher.

These steps depend on each other's outputs. Changing one usually requires revisiting the ones after it.

### Evaluation and checkpoints

`evaluate_all_tasks` produces eight accuracy matrices (head, NCM, ensemble, hierarchical; top-1 and top-5) plus task-routing accuracy. `accuracy_matrix` mirrors the hierarchical matrix. The hierarchical predictor computes `P(c | task k)` as `P_ens(c) / P_marginal(k)`, an exact Bayes conditional. Do not replace that division with a softmax; the double-softmax bug it fixed cost 17 points of top-5 accuracy and `tests/test_hierarchical_bayes.py` guards it.

Each task writes `mavis_task_{t}.pth`, `mavis_ewc_task_{t}.pth` (only when EWC is enabled), and `accuracy_matrix*.json` under the checkpoint dir. `train_mavis` auto-resumes from the highest completed task. If all 10 tasks are complete, it archives the directory with a timestamp and restarts from Task 1. `results/checkpoints/` is gitignored, so resume state is local-only.

### HPC-driven constraints (do not revert)

- `main.py` sets `TRITON_DISABLE=1` at import time.
- `models/attention.py` uses a hand-rolled multi-head attention forced to FP32 with 3D `torch.bmm` instead of `nn.MultiheadAttention` or SDPA.
- `data/tiny_imagenet.py` forces `num_workers=0` and preloads every image into RAM.

These exist because of Triton compiler and multiprocessing crashes on the SLURM cluster. They look like inefficiencies locally but are intentional.

### Config gotchas

- `MAVISConfig.load()` silently drops YAML keys that are not dataclass fields.
- `configs/mavis.yaml` contains a Windows absolute `data_dir`; `load()` falls back to the project data dir if that path is missing.
- `EncoderConfig.channels` and `use_batch_norm` are not consumed by `CNNEncoder`, which hardcodes 128/256/512/1024 channels with GroupNorm. The README's "VGG-style / BatchNorm" description is stale.
- The default `ewc_lambda = 5000` is calibrated for M4, where distillation losses are also active. The logged M2 results used `--ewc-lambda 50000`; running `train_ewc` with defaults will not reproduce them.
