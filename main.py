import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
MAVIS — Memory Augmented Visual Incremental System
====================================================
Main entry point for the project.

Usage:
    python main.py setup          # Verify environment + download dataset
    python main.py verify         # Run verification checks
    python main.py visualize      # Visualize task samples
    python main.py train_baseline # Train CNN baseline (Phase 2)
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# Project imports
from config import get_default_config, MAVISConfig, PROJECT_ROOT, CONFIGS_DIR, RESULTS_DIR


def setup_environment(config: MAVISConfig) -> dict:
    """
    Phase 1 Step 1–4: Verify environment, record metadata.
    
    Returns:
        Dictionary with environment information.
    """
    print("=" * 60)
    print("MAVIS Environment Setup")
    print("=" * 60)

    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
    }

    # TensorFlow
    try:
        import tensorflow as tf
        env_info["tensorflow_version"] = tf.__version__
        
        # GPU check
        gpus = tf.config.list_physical_devices("GPU")
        env_info["gpu_devices"] = [g.name for g in gpus]
        env_info["gpu_available"] = len(gpus) > 0
        
        if gpus:
            print(f"  [OK] TensorFlow {tf.__version__}")
            print(f"  [OK] GPU(s) found: {[g.name for g in gpus]}")
            
            # Enable memory growth to avoid grabbing all VRAM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("  [OK] GPU memory growth enabled")
            
            # Quick GPU computation test
            with tf.device("/GPU:0"):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                start = time.perf_counter()
                c = tf.matmul(a, b)
                _ = c.numpy()  # Force execution
                gpu_time = time.perf_counter() - start
                env_info["gpu_matmul_time"] = f"{gpu_time*1000:.1f}ms"
                print(f"  [OK] GPU matmul test: {gpu_time*1000:.1f}ms")
        else:
            print("  [!!] No GPU found! Training will be very slow.")
            
        # Mixed precision
        if config.training.mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                env_info["mixed_precision"] = "mixed_float16"
                print("  [OK] Mixed precision: FP16 enabled")
            except Exception as e:
                env_info["mixed_precision"] = f"failed: {e}"
                print(f"  [!!] Mixed precision failed: {e}")
                
    except ImportError:
        print("  [!!] TensorFlow not installed!")
        env_info["tensorflow_version"] = "NOT INSTALLED"

    # CUDA info (best effort)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            env_info["nvidia_gpu_info"] = gpu_info
            print(f"  [OK] NVIDIA: {gpu_info}")
    except FileNotFoundError:
        pass

    # Save environment info
    env_path = os.path.join(RESULTS_DIR, "raw", "environment.json")
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2, default=str)
    print(f"\n  Saved environment info → {env_path}")

    return env_info


def setup_dataset(config: MAVISConfig) -> None:
    """Phase 1 Step 5–7: Download dataset, create task generator, visualize."""
    from data.download import download_tiny_imagenet, verify_dataset

    print("\n" + "=" * 60)
    print("Dataset Setup")
    print("=" * 60)

    # Download
    dataset_dir = download_tiny_imagenet(os.path.dirname(config.dataset.data_dir))

    # Verify
    info = verify_dataset(dataset_dir)
    print(f"\n  Train: {info['train_classes']} classes, {info['train_images']} images")
    print(f"  Val:   {info['val_classes']} classes, {info['val_images']} images")
    print(f"  Sample: {info['sample_dimensions']}")
    if info["issues"]:
        print(f"  [WARN] Issues: {info['issues']}")

    # Task generator
    from data.task_generator import TaskGenerator
    task_gen = TaskGenerator(
        data_dir=config.dataset.data_dir,
        num_tasks=config.dataset.num_tasks,
        classes_per_task=config.dataset.classes_per_task,
        seed=config.dataset.seed,
    )
    print(f"\n{task_gen.summary()}")
    task_gen.validate()
    task_gen.save_definitions(os.path.join(CONFIGS_DIR, "task_definitions.json"))


def visualize_samples(config: MAVISConfig) -> None:
    """Phase 1 Step 8: Visualize samples from each task."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    from data.task_generator import TaskGenerator
    from data.tiny_imagenet import TinyImageNetLoader

    task_gen = TaskGenerator(
        data_dir=config.dataset.data_dir,
        num_tasks=config.dataset.num_tasks,
        classes_per_task=config.dataset.classes_per_task,
        seed=config.dataset.seed,
    )

    loader = TinyImageNetLoader(
        data_dir=config.dataset.data_dir,
        task_generator=task_gen,
        batch_size=16,
        image_size=config.dataset.image_size,
        seed=config.dataset.seed,
    )

    # Create a grid: one row per task, 8 sample images per row
    fig, axes = plt.subplots(
        config.dataset.num_tasks, 8,
        figsize=(20, 2.5 * config.dataset.num_tasks)
    )
    fig.suptitle("MAVIS — Sample Images per Task", fontsize=16, fontweight="bold")

    for task_id in range(1, config.dataset.num_tasks + 1):
        ds = loader.get_task_train_dataset(task_id, augment=False)
        for images, labels in ds.take(1):
            for j in range(min(8, images.shape[0])):
                ax = axes[task_id - 1, j]
                ax.imshow(images[j].numpy())
                ax.set_title(f"L:{labels[j].numpy()}", fontsize=8)
                ax.axis("off")
                if j == 0:
                    ax.set_ylabel(f"Task {task_id}", fontsize=10, rotation=0, 
                                 labelpad=40, va="center")

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "figures", "task_samples.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Visualize] Saved task sample grid → {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MAVIS — Memory Augmented Visual Incremental System")
    parser.add_argument(
        "command",
        choices=["setup", "verify", "visualize", "config"],
        help="Command to run",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config = MAVISConfig.load(args.config)
    else:
        config = get_default_config()

    if args.command == "config":
        print(config.summary())
        config.save(os.path.join(CONFIGS_DIR, "mavis.yaml"))

    elif args.command == "setup":
        setup_environment(config)
        setup_dataset(config)

    elif args.command == "verify":
        setup_environment(config)

    elif args.command == "visualize":
        visualize_samples(config)


if __name__ == "__main__":
    main()
