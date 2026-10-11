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

## Phase 4 Optimization: Intra-Task Plasticity & Inter-Task Stability Levers
**Goal:** Boost both individual task accuracies (training & validation: 72% -> 85%+) and continual memory retention across all tasks (raising retention from ~30% to 50%+).

### 1. Lever 1: Multi-Prototype Memory Bank (K-Means Clustering)
- **Files:** `models/memory.py`, `models/mavis.py`
- Replaced single-prototype averaging per class with pure PyTorch K-Means clustering ($K=3$ prototypes per class).
- Memory Bank tensor shape expanded to `[num_tasks, classes_per_task, K, feature_dim]`.
- Implemented `compute_ncm_scores()` using maximum cosine similarity over the $K$ sub-cluster prototypes per class for fine-grained multi-modal representation matching.

### 2. Lever 2: Balanced Classifier Head via Weight Aligning (WA - CVPR 2020)
- **Files:** `models/classifier.py`, `continual/mavis_trainer.py`
- Added `task_scale` buffer in `CosineLinear` and implemented `align_weights()` in `IncrementalClassifier`.
- Calibrates new class weights against historical class weights using norm ratios and prototype alignment ratios to eliminate the classifier recency bias.

### 3. Lever 3: Calibrated Augmentations for 64×64 Tiny ImageNet
- **File:** `data/augmentation.py`
- Removed harsh 224×224 ImageNet `AutoAugment` policy and lowered `RandomErasing` probability to `0.2` with smaller erase scale `(0.02, 0.20)`.
- Preserved `RandomCrop(64, padding=4)` and `RandomHorizontalFlip()`, preventing visual artifact destruction on 64×64 resolution.

### 4. Lever 4: Multi-Step Distillation Anchoring & Prototype Alignment Loss
- **Files:** `continual/mavis_trainer.py`, `config.py`, `main.py`
- Increased representation distillation weight $\lambda_{distill}$ to `4.0` to counter cumulative drift across sequential tasks.
- Introduced Prototype Alignment Loss $\mathcal{L}_{proto} = 1.0 - \cos(\mathbf{f}_{curr}(x), \mathbf{P}_y)$ during task training, pulling representations tightly into cluster centroids and anchoring the feature space coordinates to the Memory Bank.
- Upgraded evaluation loop to report both Weight-Aligned linear head accuracy and Multi-Prototype NCM accuracy.

