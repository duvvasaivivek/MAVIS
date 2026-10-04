import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
MAVIS -- Memory Augmented Visual Incremental System (PyTorch)
=============================================================
Main entry point for the project.

Usage:
    python main.py setup          # Verify environment + download dataset
    python main.py verify         # Run verification checks
    python main.py visualize      # Visualize task samples
    python main.py train_baseline # Train PyTorch baseline (Phase 2)
"""

import os
import json
import argparse
import torch

# Project imports
from config import get_default_config, MAVISConfig, PROJECT_ROOT, CONFIGS_DIR, RESULTS_DIR
from data.task_generator import TaskGenerator


def setup_environment(config: MAVISConfig):
    """Initialize system, verify PyTorch/CUDA, and download data."""
    from data.download import download_tiny_imagenet

    print("=" * 60)
    print("MAVIS Environment Setup (PyTorch)")
    print("=" * 60)

    # 1. Check PyTorch and CUDA
    print(f"  [OK] PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  [OK] GPU found: {gpu_name}")
        device = "cuda"
    else:
        print("  [WARN] No GPU found. Running on CPU.")
        device = "cpu"

    # 2. Check/Download Dataset
    if not os.path.exists(config.dataset.data_dir):
        print(f"  [!!] Dataset not found at {config.dataset.data_dir}.")
        download_tiny_imagenet(config.dataset.data_dir)
    else:
        print(f"  [OK] Dataset ready at {config.dataset.data_dir}")

    return device


def visualize_samples(config: MAVISConfig):
    """Visualize a grid of task samples using matplotlib."""
    print("[Visualize] Not yet implemented for PyTorch pipeline.")


def train_baseline(config: MAVISConfig):
    """
    Phase 2: Baseline Model Training loop (No memory, no EWC).
    Trains sequentially across all tasks and measures catastrophic forgetting.
    """
    from data.tiny_imagenet import TinyImageNetLoader
    from models.baseline import BaselineCNN
    from continual.trainer import ContinualTrainer

    print("\n" + "=" * 60)
    print("Phase 2: Training Baseline CNN (PyTorch)")
    print("=" * 60)

    device = setup_environment(config)

    # 1. Task Generator
    task_gen = TaskGenerator(
        data_dir=config.dataset.data_dir,
        num_tasks=config.dataset.num_tasks,
        classes_per_task=config.dataset.classes_per_task,
        seed=config.dataset.seed
    )
    loader = TinyImageNetLoader(
        data_dir=config.dataset.data_dir,
        task_generator=task_gen,
        batch_size=config.training.batch_size,
        num_workers=2,
        aug_config=config.augmentation
    )

    # 2. Model
    model = BaselineCNN(
        feature_dim=config.encoder.feature_dim,
        total_classes=task_gen.num_classes
    )

    # 3. Continual Trainer
    trainer = ContinualTrainer(
        model=model,
        loader=loader,
        task_gen=task_gen,
        learning_rate=config.training.learning_rate,
        epochs_per_task=config.training.epochs_per_task,
        checkpoint_dir=os.path.join(RESULTS_DIR, "checkpoints", "baseline"),
        device=device
    )

    # 4. Training Loop (Sequential Tasks)
    for task_id in range(config.dataset.num_tasks):
        trainer.train_task(task_id)
        trainer.evaluate_all_tasks(task_id)
        forgetting = trainer.compute_forgetting()
        print(f"  -> Average Forgetting so far: {forgetting:.4f}")

    # 5. Save Final Results
    final_res = {
        "final_accuracy_matrix": trainer.accuracy_matrix,
        "final_forgetting": trainer.compute_forgetting()
    }
    res_path = os.path.join(RESULTS_DIR, "raw", "baseline_results.json")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(final_res, f, indent=2)
    print(f"\n[Baseline] Saved metrics to {res_path}")


def train_ewc(config: MAVISConfig):
    """
    Phase 3: EWC Model Training loop.
    Same architecture as baseline but with Fisher-based weight regularization.
    """
    from data.tiny_imagenet import TinyImageNetLoader
    from models.baseline import BaselineCNN
    from continual.trainer import ContinualTrainer

    print("\n" + "=" * 60)
    print(f"Phase 3: Training CNN + EWC (lambda={config.ewc.ewc_lambda})")
    print("=" * 60)

    device = setup_environment(config)

    task_gen = TaskGenerator(
        data_dir=config.dataset.data_dir,
        num_tasks=config.dataset.num_tasks,
        classes_per_task=config.dataset.classes_per_task,
        seed=config.dataset.seed
    )
    loader = TinyImageNetLoader(
        data_dir=config.dataset.data_dir,
        task_generator=task_gen,
        batch_size=config.training.batch_size,
        num_workers=2,
        aug_config=config.augmentation
    )

    model = BaselineCNN(
        feature_dim=config.encoder.feature_dim,
        total_classes=task_gen.num_classes
    )

    trainer = ContinualTrainer(
        model=model,
        loader=loader,
        task_gen=task_gen,
        learning_rate=config.training.learning_rate,
        epochs_per_task=config.training.epochs_per_task,
        checkpoint_dir=os.path.join(RESULTS_DIR, "checkpoints", "ewc"),
        device=device,
        ewc_lambda=config.ewc.ewc_lambda,
        fisher_samples=config.ewc.fisher_samples
    )

    for task_id in range(config.dataset.num_tasks):
        trainer.train_task(task_id)
        trainer.evaluate_all_tasks(task_id)
        forgetting = trainer.compute_forgetting()
        print(f"  -> Average Forgetting so far: {forgetting:.4f}")

    final_res = {
        "ewc_lambda": config.ewc.ewc_lambda,
        "fisher_samples": config.ewc.fisher_samples,
        "final_accuracy_matrix": trainer.accuracy_matrix,
        "final_forgetting": trainer.compute_forgetting()
    }
    res_path = os.path.join(RESULTS_DIR, "raw", "ewc_results.json")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(final_res, f, indent=2)
    print(f"\n[EWC] Saved metrics to {res_path}")


def train_mavis(config: MAVISConfig):
    """
    Phase 4 & 5: MAVIS Complete Architecture.
    Uses the external Neural Memory Bank and Multi-Head Attention.
    """
    from data.tiny_imagenet import TinyImageNetLoader
    from models.mavis import MAVIS
    from continual.mavis_trainer import MAVISTrainer

    print("\n" + "=" * 60)
    print("Phase 4 & 5: Training MAVIS (Memory + Attention)")
    print("=" * 60)

    device = setup_environment(config)

    task_gen = TaskGenerator(
        data_dir=config.dataset.data_dir,
        num_tasks=config.dataset.num_tasks,
        classes_per_task=config.dataset.classes_per_task,
        seed=config.dataset.seed
    )
    loader = TinyImageNetLoader(
        data_dir=config.dataset.data_dir,
        task_generator=task_gen,
        batch_size=config.training.batch_size,
        num_workers=2,
        aug_config=config.augmentation
    )

    model = MAVIS(config)

    trainer = MAVISTrainer(
        model=model,
        loader=loader,
        task_gen=task_gen,
        learning_rate=config.training.learning_rate,
        epochs_per_task=config.training.epochs_per_task,
        checkpoint_dir=os.path.join(RESULTS_DIR, "checkpoints", "mavis"),
        device=device
    )

    for task_id in range(config.dataset.num_tasks):
        trainer.train_task(task_id)
        trainer.evaluate_all_tasks(task_id)
        forgetting = trainer.compute_forgetting()
        print(f"  -> Average Forgetting so far: {forgetting:.4f}")

    final_res = {
        "final_accuracy_matrix": trainer.accuracy_matrix,
        "final_forgetting": trainer.compute_forgetting()
    }
    res_path = os.path.join(RESULTS_DIR, "raw", "mavis_results.json")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(final_res, f, indent=2)
    print(f"\n[MAVIS] Saved metrics to {res_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MAVIS -- Memory Augmented Visual Incremental System"
    )
    parser.add_argument(
        "command",
        choices=["setup", "verify", "visualize", "config", "train_baseline", "train_ewc", "train_mavis"],
        help="Command to run",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config = MAVISConfig.load(args.config)
    else:
        config = get_default_config()

    # Route command
    if args.command == "config":
        print(config.summary())
    elif args.command in ("setup", "verify"):
        setup_environment(config)
    elif args.command == "visualize":
        visualize_samples(config)
    elif args.command == "train_baseline":
        train_baseline(config)
    elif args.command == "train_ewc":
        train_ewc(config)
    elif args.command == "train_mavis":
        train_mavis(config)


if __name__ == "__main__":
    main()
