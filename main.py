import sys
import io
import os
os.environ["TRITON_DISABLE"] = "1"  # CRITICAL: Completely disable Triton to prevent C++ compiler crashes on HPC

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

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
import torch.multiprocessing as mp

# In-process data loading with RAM caching eliminates all Python 3.14 multiprocessing / sem_unlink bugs

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

    # 1.5 Enable cuDNN benchmarking for faster convolutions
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  [OK] cuDNN benchmark enabled for max speed")

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
        num_workers=0,
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
        num_workers=0,
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


def train_mavis(config: MAVISConfig, resume: bool = True):
    """
    Phase 4 & 5: MAVIS Complete Architecture.
    Uses the external Neural Memory Bank and Multi-Head Attention.
    """
    from data.tiny_imagenet import TinyImageNetLoader
    from models.mavis import MAVIS
    from continual.mavis_trainer import MAVISTrainer

    print("\n" + "=" * 60)
    mode_str = f"Online EWC λ={config.ewc.ewc_lambda}" if config.ewc.enabled else "EWC Disabled"
    print(f"Phase 5: Training Full MAVIS (Memory + Attention + {mode_str})")
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
        num_workers=0,  # num_workers=0 with RAM caching eliminates multiprocessing IPC crashes in Python 3.14
        aug_config=config.augmentation
    )

    model = MAVIS(config)

    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints", "mavis")
    ewc_lambda = config.ewc.ewc_lambda if config.ewc.enabled else 0.0
    trainer = MAVISTrainer(
        model=model,
        loader=loader,
        task_gen=task_gen,
        learning_rate=config.training.learning_rate,
        epochs_per_task=config.training.epochs_per_task,
        checkpoint_dir=checkpoint_dir,
        device=device,
        distill_lambda=getattr(config.training, "distill_lambda", 3.0),
        spatial_distill_lambda=getattr(config.training, "spatial_distill_lambda", 1.5),
        proto_lambda=getattr(config.training, "proto_lambda", 0.5),
        repel_lambda=getattr(config.training, "repel_lambda", 2.0),
        lr_backbone_factor=getattr(config.training, "lr_backbone_factor", 0.20),
        label_smoothing=getattr(config.training, "label_smoothing", 0.02),
        proto_replay_lambda=getattr(config.training, "proto_replay_lambda", 2.0),
        proto_finetune_epochs=getattr(config.training, "proto_finetune_epochs", 25),
        proto_finetune_draws=getattr(config.training, "proto_finetune_draws", 10),
        ewc_lambda=ewc_lambda,
        fisher_samples=config.ewc.fisher_samples,
        gamma=getattr(config.ewc, "gamma", 0.95),
    )

    # Auto-resume from latest completed task if checkpoint exists
    import copy
    start_task = 0
    if not resume:
        print(f"[MAVIS] Starting fresh training run (--fresh specified). Cleaning previous checkpoints in {checkpoint_dir}...\n", flush=True)
        if os.path.exists(checkpoint_dir):
            for fname in os.listdir(checkpoint_dir):
                if fname.startswith("mavis_") or fname.startswith("accuracy_matrix"):
                    try:
                        os.remove(os.path.join(checkpoint_dir, fname))
                    except Exception:
                        pass
    else:
        for t in range(config.dataset.num_tasks, 0, -1):
            ckpt_path = os.path.join(checkpoint_dir, f"mavis_task_{t}.pth")
            if os.path.exists(ckpt_path):
                print(f"\n[MAVIS] Found checkpoint for completed Task {t} at {ckpt_path}. Checking compatibility...")
                try:
                    state_dict = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(state_dict, strict=True)
                    trainer.old_encoder = copy.deepcopy(model.encoder)
                    trainer.old_encoder.eval()
                    for p in trainer.old_encoder.parameters():
                        p.requires_grad = False
                    
                    # Restore base anchor encoder E_0 from Task 1 checkpoint
                    base_ckpt = os.path.join(checkpoint_dir, "mavis_task_1.pth")
                    if os.path.exists(base_ckpt):
                        try:
                            base_state = torch.load(base_ckpt, map_location=device)
                            base_model = MAVIS(config).to(device)
                            base_model.load_state_dict(base_state, strict=True)
                            trainer.base_encoder = copy.deepcopy(base_model.encoder)
                            trainer.base_encoder.eval()
                            for p in trainer.base_encoder.parameters():
                                p.requires_grad = False
                            print("[MAVIS] Successfully loaded Base Anchor Encoder E_0 from Task 1.")
                        except Exception:
                            pass
                    
                    metrics_ckpt = os.path.join(checkpoint_dir, "accuracy_matrix.json")
                    if os.path.exists(metrics_ckpt):
                        with open(metrics_ckpt, "r") as f:
                            trainer.accuracy_matrix = json.load(f)[:t]
                    metrics_hier_ckpt = os.path.join(checkpoint_dir, "accuracy_matrix_hier.json")
                    if os.path.exists(metrics_hier_ckpt):
                        with open(metrics_hier_ckpt, "r") as f:
                            trainer.accuracy_matrix_hier = json.load(f)[:t]
                    metrics_ncm_ckpt = os.path.join(checkpoint_dir, "accuracy_matrix_ncm.json")
                    if os.path.exists(metrics_ncm_ckpt):
                        with open(metrics_ncm_ckpt, "r") as f:
                            trainer.accuracy_matrix_ncm = json.load(f)[:t]
                    metrics_head_ckpt = os.path.join(checkpoint_dir, "accuracy_matrix_head.json")
                    if os.path.exists(metrics_head_ckpt):
                        with open(metrics_head_ckpt, "r") as f:
                            trainer.accuracy_matrix_head = json.load(f)[:t]
                    metrics_ens_ckpt = os.path.join(checkpoint_dir, "accuracy_matrix_ens.json")
                    if os.path.exists(metrics_ens_ckpt):
                        with open(metrics_ens_ckpt, "r") as f:
                            trainer.accuracy_matrix_ens = json.load(f)[:t]
                    metrics_top5_ens_ckpt = os.path.join(checkpoint_dir, "accuracy_matrix_top5_ens.json")
                    if os.path.exists(metrics_top5_ens_ckpt):
                        with open(metrics_top5_ens_ckpt, "r") as f:
                            trainer.accuracy_matrix_top5_ens = json.load(f)[:t]
                    metrics_top5_head_ckpt = os.path.join(checkpoint_dir, "accuracy_matrix_top5_head.json")
                    if os.path.exists(metrics_top5_head_ckpt):
                        with open(metrics_top5_head_ckpt, "r") as f:
                            trainer.accuracy_matrix_top5_head = json.load(f)[:t]
                    metrics_top5_hier_ckpt = os.path.join(checkpoint_dir, "accuracy_matrix_top5_hier.json")
                    if os.path.exists(metrics_top5_hier_ckpt):
                        with open(metrics_top5_hier_ckpt, "r") as f:
                            trainer.accuracy_matrix_top5_hier = json.load(f)[:t]
                    
                    # Restore Online EWC Fisher state from completed Task t
                    ewc_ckpt_path = os.path.join(checkpoint_dir, f"mavis_ewc_task_{t}.pth")
                    if os.path.exists(ewc_ckpt_path) and trainer.ewc_module is not None:
                        try:
                            ewc_state = torch.load(ewc_ckpt_path, map_location=device)
                            trainer.ewc_module.load_state_dict(ewc_state)
                            print(f"[MAVIS] Successfully loaded Online EWC state from Task {t}.", flush=True)
                        except Exception as e:
                            print(f"[MAVIS] Note on loading EWC state: {e}", flush=True)
                    
                    start_task = t
                    if start_task >= config.dataset.num_tasks:
                        print(f"\n[MAVIS] All {config.dataset.num_tasks} tasks already completed in {checkpoint_dir}.")
                        print(f"[MAVIS] Archiving previous completed run and auto-starting fresh training run from Task 1.\n", flush=True)
                        import shutil, datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        archive_dir = f"{checkpoint_dir}_completed_{timestamp}"
                        try:
                            shutil.copytree(checkpoint_dir, archive_dir)
                            for fname in os.listdir(checkpoint_dir):
                                fpath = os.path.join(checkpoint_dir, fname)
                                if os.path.isfile(fpath):
                                    os.remove(fpath)
                            print(f"[MAVIS] Successfully archived previous checkpoints to {archive_dir}.\n", flush=True)
                        except Exception as e:
                            print(f"[MAVIS] Note on archiving: {e}\n", flush=True)
                        start_task = 0
                        trainer.old_encoder = None
                        trainer.base_encoder = None
                        trainer.accuracy_matrix = []
                        trainer.accuracy_matrix_hier = []
                        trainer.accuracy_matrix_ncm = []
                        trainer.accuracy_matrix_head = []
                        trainer.accuracy_matrix_ens = []
                        trainer.accuracy_matrix_top5_ens = []
                        trainer.accuracy_matrix_top5_head = []
                        trainer.accuracy_matrix_top5_hier = []
                        if trainer.ewc_module is not None:
                            trainer.ewc_module.running_fisher = None
                            trainer.ewc_module.saved_params = None
                            trainer.ewc_module.task_count = 0
                        model = MAVIS(config).to(device)
                        trainer.model = model
                        break

                    print(f"[MAVIS] Successfully resumed! Continuing from Task {start_task + 1}...\n", flush=True)
                    break
                except RuntimeError as e:
                    print(f"[MAVIS] Checkpoint {ckpt_path} was saved from an older architecture and is incompatible:")
                    print(f"        {e}")
                    print(f"[MAVIS] Starting a clean training run from Task 1.\n")
                    start_task = 0
                    if trainer.ewc_module is not None:
                        trainer.ewc_module.running_fisher = None
                        trainer.ewc_module.saved_params = None
                        trainer.ewc_module.task_count = 0
                    break

    for task_id in range(start_task, config.dataset.num_tasks):
        trainer.train_task(task_id)
        trainer.evaluate_all_tasks(task_id)
        forgetting_hier = trainer.compute_forgetting(trainer.accuracy_matrix_hier)
        forgetting_ens = trainer.compute_forgetting(trainer.accuracy_matrix_ens)
        forgetting_ens_top5 = trainer.compute_forgetting(trainer.accuracy_matrix_top5_ens)
        forgetting_ncm = trainer.compute_forgetting(trainer.accuracy_matrix_ncm)
        forgetting_head = trainer.compute_forgetting(trainer.accuracy_matrix_head)
        forgetting_head_top5 = trainer.compute_forgetting(trainer.accuracy_matrix_top5_head)
        print(f"  -> Average Forgetting so far | Hierarchical: {forgetting_hier:.4f} | Ensemble: {forgetting_ens:.4f} (Top-5: {forgetting_ens_top5:.4f}) | Head: {forgetting_head:.4f} (Top-5: {forgetting_head_top5:.4f}) | NCM: {forgetting_ncm:.4f}", flush=True)

    final_res = {
        "ewc_enabled": config.ewc.enabled,
        "ewc_lambda": ewc_lambda,
        "gamma": getattr(config.ewc, "gamma", 0.95),
        "fisher_samples": config.ewc.fisher_samples,
        "final_accuracy_matrix": trainer.accuracy_matrix,
        "final_accuracy_matrix_hier": trainer.accuracy_matrix_hier,
        "final_accuracy_matrix_ens": trainer.accuracy_matrix_ens,
        "final_accuracy_matrix_ncm": trainer.accuracy_matrix_ncm,
        "final_accuracy_matrix_head": trainer.accuracy_matrix_head,
        "final_accuracy_matrix_top5_ens": trainer.accuracy_matrix_top5_ens,
        "final_accuracy_matrix_top5_head": trainer.accuracy_matrix_top5_head,
        "final_accuracy_matrix_top5_hier": trainer.accuracy_matrix_top5_hier,
        "final_forgetting_hier": trainer.compute_forgetting(trainer.accuracy_matrix_hier),
        "final_forgetting_ens": trainer.compute_forgetting(trainer.accuracy_matrix_ens),
        "final_forgetting_ens_top5": trainer.compute_forgetting(trainer.accuracy_matrix_top5_ens),
        "final_forgetting_ncm": trainer.compute_forgetting(trainer.accuracy_matrix_ncm),
        "final_forgetting_head": trainer.compute_forgetting(trainer.accuracy_matrix_head),
        "final_forgetting_head_top5": trainer.compute_forgetting(trainer.accuracy_matrix_top5_head),
        "final_forgetting_hier_top5": trainer.compute_forgetting(trainer.accuracy_matrix_top5_hier)
    }
    res_path = os.path.join(RESULTS_DIR, "raw", "mavis_results.json")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(final_res, f, indent=2)
    print(f"\n[MAVIS] Saved metrics to {res_path}", flush=True)


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
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per task (e.g. 50)")
    parser.add_argument("--ewc-lambda", type=float, default=None, help="Override EWC lambda regularization strength (e.g. 2000.0)")
    parser.add_argument("--no-ewc", action="store_true", help="Disable EWC regularization in MAVIS (ablation model M3)")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh from Task 1 even if checkpoints exist")
    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config = MAVISConfig.load(args.config)
    else:
        config = get_default_config()

    if args.epochs is not None:
        config.training.epochs_per_task = args.epochs
        print(f"[Config] Command-line override: epochs_per_task = {args.epochs}")

    if args.ewc_lambda is not None:
        config.ewc.ewc_lambda = args.ewc_lambda
        print(f"[Config] Command-line override: ewc_lambda = {args.ewc_lambda}")

    if args.no_ewc:
        config.ewc.enabled = False
        print(f"[Config] Command-line override: EWC regularization DISABLED")

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
        train_mavis(config, resume=not args.fresh)


if __name__ == "__main__":
    main()
