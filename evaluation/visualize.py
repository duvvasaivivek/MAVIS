"""
MAVIS Visualization Module
==========================
Publication-quality plotting functions for continual learning:
1. Task sample grids.
2. Accuracy curves across sequential tasks.
3. Forgetting comparison bar charts.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict, List, Optional


def visualize_task_samples(config, output_path: Optional[str] = None) -> str:
    """
    Renders a representative grid of Tiny ImageNet images (1 per task or 1 per class).
    Saves to results/figures/task_samples.png.
    """
    from data.task_generator import TaskGenerator

    if output_path is None:
        from config import RESULTS_DIR
        output_path = os.path.join(RESULTS_DIR, "figures", "task_samples.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    task_gen = TaskGenerator(
        data_dir=config.dataset.data_dir,
        num_tasks=config.dataset.num_tasks,
        classes_per_task=config.dataset.classes_per_task,
        seed=config.dataset.seed
    )

    num_tasks = min(config.dataset.num_tasks, 10)
    samples_per_task = 5

    fig, axes = plt.subplots(num_tasks, samples_per_task, figsize=(10, 2 * num_tasks))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    train_dir = os.path.join(config.dataset.data_dir, "train")

    for t in range(num_tasks):
        task_def = task_gen.get_task(t + 1)
        classes = task_def.class_names[:samples_per_task]
        for c_idx, class_name in enumerate(classes):
            ax = axes[t, c_idx] if num_tasks > 1 else axes[c_idx]
            class_img_dir = os.path.join(train_dir, class_name, "images")
            if os.path.exists(class_img_dir):
                img_files = [f for f in os.listdir(class_img_dir) if f.endswith(".JPEG")]
                if img_files:
                    img_path = os.path.join(class_img_dir, img_files[0])
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
            ax.axis("off")
            if c_idx == 0:
                ax.set_title(f"Task {t+1}", fontsize=10, loc="left", fontweight="bold")

    fig.suptitle("MAVIS Continual Task Splits (Tiny ImageNet)", fontsize=13, y=0.99, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved task samples grid to {output_path}")
    return output_path


def plot_accuracy_curves(runs: Dict[str, List[List[float]]], output_path: str) -> None:
    """
    Plots the average accuracy curve across continual tasks for multiple methods.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))

    for name, matrix in runs.items():
        avg_accs = [float(np.mean(row)) * 100 for row in matrix]
        tasks = list(range(1, len(avg_accs) + 1))
        plt.plot(tasks, avg_accs, marker="o", linewidth=2, label=name)

    plt.title("Continual Learning Average Accuracy across Tasks", fontsize=12, fontweight="bold")
    plt.xlabel("Task ID", fontsize=11)
    plt.ylabel("Average Accuracy (%)", fontsize=11)
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Visualize] Saved accuracy curves to {output_path}")
