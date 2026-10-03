# MAVIS Project Implementation Log

This document serves as a living record of architectural decisions, implementation details, bug fixes, and progress across all phases of the MAVIS project.

---

## Phase 1: Foundation & Infrastructure (Completed)
**Date:** September 2026
**Goal:** Establish the project environment, configure hyperparameter management, and build robust data pipelines for the Tiny ImageNet dataset.

### 1. Environment & Dependencies
- **Virtual Environment:** Set up a local Python virtual environment (`.venv`).
- **TensorFlow Downgrade:** Initially attempted to install `tensorflow==2.20` + CUDA pip wheels. However, due to Windows pip resolution issues and lack of local CUDA 11 toolkit, we downgraded to `tensorflow==2.10.1` and `numpy<2.0.0` (v1.26.4). TF 2.10 is the last version to natively support GPU on Windows without WSL2. 
- **Current GPU Status:** The system relies on CPU execution for now, but the codebase is fully prepared for GPU execution (mixed precision FP16 is enabled and handled securely).
- **Console Fix:** Applied a UTF-8 stdout wrapper in `main.py` to prevent Windows `cp1252` encoding errors when printing UI symbols (✓, ✗, ⚠).

### 2. Configuration Management
- **File:** `config.py`
- **Details:** Built a central, strongly-typed configuration system using Python `dataclasses`.
- Supports dynamic loading/saving via YAML.
- Covers distinct config categories: `DatasetConfig`, `TrainingConfig`, `EWCConfig`, `MemoryConfig`, and `ArchitectureConfig`.

### 3. Data Pipeline & Task Generator
- **Files:** `data/download.py`, `data/task_generator.py`, `data/augmentation.py`, `data/tiny_imagenet.py`
- **Details:**
  - **Download:** Automated the download and extraction of the 237MB Tiny ImageNet dataset.
  - **Validation Setup:** Wrote a parser to reorganize the flat validation images into class-specific subdirectories matching the training set format. *Fix applied:* Added a retry mechanism with backoff for `shutil.move` to circumvent Windows file locking issues during extraction.
  - **Task Generation:** Deterministically split the 200 classes into **10 sequential tasks** (20 classes each). Fixed seed `42` is used, and the definitions are serialized to JSON for strict reproducibility.
  - **Augmentation:** Applied `tf.image` augmentations (random flip, random crop with 4px padding, and color jitter) identically for training. Validation uses center crop.
  - **tf.data.Dataset:** Built a highly efficient lazy-loading pipeline using `tf.io.decode_jpeg` and `tf.data.AUTOTUNE` to prevent CPU bottlenecking. Included cumulative validation set logic for tracking catastrophic forgetting.

### 4. Entry Point & Visualization
- **File:** `main.py`
- **Details:** Added CLI commands `setup`, `verify`, `config`, and `visualize`.
- Successfully generated a grid of sample images (`results/figures/task_samples.png`), pulling 1 batch from each of the 10 isolated tasks to prove the continual-learning split works correctly.

---

## Phase 2: Baseline CNN (In Progress)
**Goal:** Build the raw CNN encoder, classifier head, and the continual learning training loop to establish a baseline for catastrophic forgetting.

*(Details will be appended as implementation progresses...)*
