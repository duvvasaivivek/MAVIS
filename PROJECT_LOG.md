# MAVIS Project Implementation Log

This document serves as a living record of architectural decisions, implementation details, bug fixes, and progress across all phases of the MAVIS project.

---

## **MAJOR ARCHITECTURAL PIVOT: PyTorch Migration**
**Date:** October 2026
**Reason:** TensorFlow 2.10.1 (the last version to natively support GPU on Windows) proved too complex with the DirectML plugin and lacked robust tooling for EWC Fisher Information Matrix calculations.
**Result:** The entire codebase (models, datasets, augmentations, and trainers) was fundamentally rewritten from TensorFlow/Keras to **PyTorch**. The environment now seamlessly supports the native RTX 4050 GPU via CUDA 12.1.

---

## Phase 1: Foundation & Infrastructure (Completed)
**Goal:** Establish the project environment, configure hyperparameter management, and build robust data pipelines for the Tiny ImageNet dataset.

### 1. Environment & Dependencies
- **Virtual Environment:** Set up a local Python virtual environment (`.venv`).
- **PyTorch Stack:** Installed `torch`, `torchvision`, and `torchaudio` pointing to `cu121` index for flawless Windows native GPU support.
- **Console Fix:** Applied a UTF-8 stdout wrapper in `main.py` to prevent Windows `cp1252` encoding errors when printing UI symbols.

### 2. Configuration Management
- **File:** `config.py`
- **Details:** Built a central, strongly-typed configuration system using Python `dataclasses`. Supports dynamic YAML loading.

### 3. Data Pipeline & Task Generator
- **Files:** `data/download.py`, `data/task_generator.py`, `data/augmentation.py`, `data/tiny_imagenet.py`
- **Validation Setup:** Wrote a parser to reorganize the flat Tiny ImageNet validation images into class-specific subdirectories. 
- **Task Generation:** Deterministically split the 200 classes into **10 sequential tasks** (20 classes each). Fixed seed `42` is used, and the definitions are serialized to JSON.
- **PyTorch DataLoader:** Replaced `tf.data` with `torch.utils.data.DataLoader` and a custom `TinyImageNetTaskDataset` using PIL and `torchvision.transforms`.

---

## Phase 2: Baseline CNN (Completed)
**Goal:** Build the raw CNN encoder, classifier head, and the continual learning training loop to establish a baseline for catastrophic forgetting.

### 1. CNN Encoder & Classifier (PyTorch)
- **Files:** `models/encoder.py`, `models/classifier.py`, `models/baseline.py`
- **Encoder:** Implemented a robust 4-block VGG-style CNN encoder using `torch.nn.Sequential` (Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D).
- **Classifier:** Created an `IncrementalClassifier` (`nn.Linear`) mapping the 512-d feature vector to 200 logits.
- **Baseline Model:** Combined both into `BaselineCNN` (Model M1).

### 2. Continual Learning Training Loop
- **File:** `continual/trainer.py`
- **Details:** 
  - Iterates over all tasks sequentially.
  - Implements AMP (Automatic Mixed Precision) via `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` for massive speedups on the RTX 4050.
  - **Metrics Tracking:** Computes "catastrophic forgetting" by evaluating the model on all historical tasks and accumulating an accuracy matrix.
  - Automatically saves PyTorch checkpoint weights (`.pth`) for each task.

### 3. Execution Integration
- **File:** `main.py`
- **Details:** Re-integrated `train_baseline`.

---

## Next Steps
- **Phase 3:** Implement Elastic Weight Consolidation (EWC) to defend against the catastrophic forgetting measured in Phase 2. This will involve computing the Fisher Information Matrix over the training data and applying an L2 penalty on critical weights.
