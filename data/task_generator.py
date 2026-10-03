"""
Continual Task Generator
=========================
Splits Tiny ImageNet (200 classes) into sequential continual-learning tasks.
Default: 10 tasks × 20 classes each.

The class order is fixed and recorded for reproducibility.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TaskDefinition:
    """A single continual-learning task."""
    task_id: int
    class_indices: List[int]       # Global class indices (0-199)
    class_names: List[str]         # WordNet IDs (e.g., 'n01443537')
    num_classes: int

    def __repr__(self) -> str:
        return (f"Task {self.task_id}: {self.num_classes} classes "
                f"[{self.class_indices[0]}..{self.class_indices[-1]}]")


class TaskGenerator:
    """
    Generates deterministic continual-learning task splits.
    
    Splits `num_classes` classes into `num_tasks` sequential groups.
    Class order is controlled by a fixed random seed for reproducibility.
    
    Args:
        data_dir: Path to tiny-imagenet-200 directory.
        num_tasks: Number of sequential tasks (default: 10).
        classes_per_task: Classes per task (default: 20).
        seed: Random seed for class ordering (default: 42).
        shuffle_classes: Whether to shuffle class order (default: True).
    """

    def __init__(
        self,
        data_dir: str,
        num_tasks: int = 10,
        classes_per_task: int = 20,
        seed: int = 42,
        shuffle_classes: bool = True,
    ):
        self.data_dir = data_dir
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.seed = seed
        self.shuffle_classes = shuffle_classes

        # Discover classes from training directory
        train_dir = os.path.join(data_dir, "train")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        # Get all class directory names (WordNet IDs) sorted for determinism
        self.all_class_names = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        self.num_classes = len(self.all_class_names)

        assert self.num_classes >= num_tasks * classes_per_task, (
            f"Not enough classes: have {self.num_classes}, "
            f"need {num_tasks * classes_per_task}"
        )

        # Build class name → index mapping
        if shuffle_classes:
            rng = np.random.RandomState(seed)
            self.class_order = rng.permutation(self.num_classes).tolist()
        else:
            self.class_order = list(range(self.num_classes))

        # class_name_to_idx: maps WordNet ID to its global integer label
        self.class_name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.all_class_names)
        }

        # Generate task definitions
        self.tasks: List[TaskDefinition] = []
        for t in range(num_tasks):
            start = t * classes_per_task
            end = start + classes_per_task
            task_class_indices = self.class_order[start:end]
            task_class_names = [self.all_class_names[i] for i in task_class_indices]
            self.tasks.append(TaskDefinition(
                task_id=t + 1,
                class_indices=task_class_indices,
                class_names=task_class_names,
                num_classes=classes_per_task,
            ))

    def get_task(self, task_id: int) -> TaskDefinition:
        """Get task by 1-indexed ID."""
        assert 1 <= task_id <= self.num_tasks, (
            f"Task ID must be 1-{self.num_tasks}, got {task_id}"
        )
        return self.tasks[task_id - 1]

    def get_classes_up_to_task(self, task_id: int) -> List[int]:
        """Get all class indices from Task 1 up to and including task_id."""
        all_classes = []
        for t in range(task_id):
            all_classes.extend(self.tasks[t].class_indices)
        return sorted(all_classes)

    def get_task_definitions(self) -> Dict[str, List[int]]:
        """
        Return task definitions as a JSON-serializable dictionary.
        
        Format: {"task_1": [0, 1, 2, ..., 19], "task_2": [20, 21, ...], ...}
        """
        return {
            f"task_{t.task_id}": t.class_indices
            for t in self.tasks
        }

    def save_definitions(self, path: str) -> None:
        """Save task definitions to JSON for reproducibility."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        definitions = {
            "metadata": {
                "num_tasks": self.num_tasks,
                "classes_per_task": self.classes_per_task,
                "total_classes": self.num_classes,
                "seed": self.seed,
                "shuffle_classes": self.shuffle_classes,
            },
            "class_order": self.class_order,
            "class_names": self.all_class_names,
            "tasks": self.get_task_definitions(),
        }
        with open(path, "w") as f:
            json.dump(definitions, f, indent=2)
        print(f"[TaskGen] Saved definitions to {path}")

    def validate(self) -> bool:
        """
        Validate task definitions: no overlap, correct counts, full coverage.
        
        Returns:
            True if all validations pass.
        """
        all_classes_seen = set()
        for task in self.tasks:
            task_set = set(task.class_indices)
            
            # Check no overlap with previous tasks
            overlap = all_classes_seen & task_set
            assert len(overlap) == 0, (
                f"Task {task.task_id} has class overlap with earlier tasks: {overlap}"
            )
            
            # Check correct number of classes
            assert task.num_classes == self.classes_per_task, (
                f"Task {task.task_id} has {task.num_classes} classes, "
                f"expected {self.classes_per_task}"
            )
            
            all_classes_seen.update(task_set)

        # Check total coverage
        expected = self.num_tasks * self.classes_per_task
        assert len(all_classes_seen) == expected, (
            f"Total unique classes: {len(all_classes_seen)}, expected {expected}"
        )

        print(f"[TaskGen] [OK] Validation passed: {self.num_tasks} tasks, "
              f"{self.classes_per_task} classes/task, "
              f"{len(all_classes_seen)} unique classes, no overlap")
        return True

    def summary(self) -> str:
        """Print a human-readable summary of all tasks."""
        lines = [
            "=" * 60,
            f"Continual Task Generator (seed={self.seed})",
            f"  {self.num_tasks} tasks × {self.classes_per_task} classes = "
            f"{self.num_tasks * self.classes_per_task} total",
            "=" * 60,
        ]
        for task in self.tasks:
            class_range = f"[{min(task.class_indices)}..{max(task.class_indices)}]"
            lines.append(f"  Task {task.task_id:2d}: classes {class_range} "
                        f"({task.num_classes} classes)")
        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_default_config, CONFIGS_DIR

    config = get_default_config()
    
    # Will fail gracefully if dataset isn't downloaded yet
    try:
        gen = TaskGenerator(
            data_dir=config.dataset.data_dir,
            num_tasks=config.dataset.num_tasks,
            classes_per_task=config.dataset.classes_per_task,
            seed=config.dataset.seed,
        )
        print(gen.summary())
        gen.validate()
        gen.save_definitions(os.path.join(CONFIGS_DIR, "task_definitions.json"))
    except FileNotFoundError as e:
        print(f"[TaskGen] Dataset not yet downloaded: {e}")
        print("[TaskGen] Run `python data/download.py` first.")
