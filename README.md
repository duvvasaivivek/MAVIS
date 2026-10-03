# MAVIS — Memory Augmented Visual Incremental System

A memory-augmented neural network for **incremental learning of new visual categories**, built from scratch with TensorFlow/Keras.

## Research Question

> Can an adaptive external neural memory, retrieved through attention and combined with parameter-importance regularization (EWC), reduce catastrophic forgetting when a network incrementally learns new visual categories under a fixed memory budget?

## Architecture

MAVIS combines two complementary knowledge-retention mechanisms:

1. **EWC (Elastic Weight Consolidation)** — A **loss-function regularizer** that penalizes changes to important CNN parameters during backpropagation. It does NOT appear in the forward data flow.

2. **Memory + Attention + Gated Fusion** — An external memory that stores old feature representations, multi-head attention for selective retrieval, and gated fusion to blend current and retrieved features. These operate in the **forward data flow**.

```
Forward Pass:    Input → CNN → Feature → [Query → Attention(Memory) → Retrieved] → Gated Fusion → Classifier
Training Loss:   L_total = L_CE + (λ/2) Σ F_i(θ_i − θ_i*)²  ← EWC penalty
```

## 4 Core Models

| Model | Description |
|---|---|
| M1: CNN | Baseline — measures raw catastrophic forgetting |
| M2: CNN + EWC | Parameter-level protection (loss regularization) |
| M3: CNN + Attention Memory | Representation-level memory (forward pass) |
| M4: MAVIS | Both mechanisms combined |

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate     # Windows
source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify environment + download dataset
python main.py setup

# Visualize task samples
python main.py visualize
```

## Dataset

- **Primary**: Tiny ImageNet (200 classes, 64×64×3, ~100K images)
- **Development**: CIFAR-100 (for fast prototyping)
- **Task Split**: 10 tasks × 20 classes (sequential)

## Hardware Target

- GPU: NVIDIA RTX 4050 (6 GB VRAM)
- Mixed precision: FP16
- Constraint: No pretrained models

## Project Structure

```
MAVIS/
├── config.py           # Central configuration
├── main.py             # Entry point
├── data/               # Dataset loading, task generation, augmentation
├── models/             # Encoder, classifier, attention, memory, fusion, MAVIS
├── continual/          # EWC, training loop, checkpoints
├── experiments/        # Experiment runners for each model
├── evaluation/         # Metrics, forgetting, plots
├── configs/            # YAML configurations
├── checkpoints/        # Saved model states per task
├── results/            # Raw results, metrics, figures, tables
└── notebooks/          # Analysis notebooks
```
