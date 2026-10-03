"""
Tiny ImageNet tf.data Pipeline
================================
Efficient TensorFlow data pipeline for loading Tiny ImageNet images
by continual-learning task.

Pipeline: Load → Parse → Normalize → Augment → Batch → Prefetch
Per spec §5.1–5.2: avoid loading full dataset into GPU memory.
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Dict

from data.augmentation import create_augmentation_fn, create_validation_fn, normalize_image
from data.task_generator import TaskGenerator, TaskDefinition


class TinyImageNetLoader:
    """
    Efficient tf.data pipeline for Tiny ImageNet with continual-learning support.
    
    Loads images from disk per-task, applies augmentation, and returns
    batched + prefetched datasets ready for training.
    
    Args:
        data_dir: Path to tiny-imagenet-200 directory.
        task_generator: TaskGenerator instance with task definitions.
        batch_size: Training batch size (default: 64).
        image_size: Image resolution (default: 64).
        seed: Random seed for shuffling (default: 42).
    """

    def __init__(
        self,
        data_dir: str,
        task_generator: TaskGenerator,
        batch_size: int = 64,
        image_size: int = 64,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.task_generator = task_generator
        self.batch_size = batch_size
        self.image_size = image_size
        self.seed = seed

        self.train_dir = os.path.join(data_dir, "train")
        self.val_dir = os.path.join(data_dir, "val")

        # Build mapping: class_name (WordNet ID) → integer index
        self.class_name_to_idx = task_generator.class_name_to_idx

        # Pre-scan all image paths and labels
        print("[Loader] Scanning dataset...")
        self.train_data = self._scan_split(self.train_dir)
        self.val_data = self._scan_split(self.val_dir)
        print(f"[Loader] Train: {len(self.train_data['paths'])} images, "
              f"Val: {len(self.val_data['paths'])} images")

    def _scan_split(self, split_dir: str) -> Dict[str, list]:
        """Scan a split directory and collect all (path, label) pairs."""
        paths = []
        labels = []

        if not os.path.exists(split_dir):
            return {"paths": paths, "labels": labels}

        for class_name in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            if class_name not in self.class_name_to_idx:
                continue

            label = self.class_name_to_idx[class_name]
            images_dir = os.path.join(class_path, "images")

            if os.path.exists(images_dir):
                for img_name in os.listdir(images_dir):
                    if img_name.lower().endswith((".jpeg", ".jpg", ".png")):
                        paths.append(os.path.join(images_dir, img_name))
                        labels.append(label)

        return {"paths": paths, "labels": labels}

    def _filter_by_classes(
        self, data: Dict[str, list], class_indices: List[int]
    ) -> Tuple[List[str], List[int]]:
        """Filter image paths and labels to only include specified classes."""
        class_set = set(class_indices)
        filtered_paths = []
        filtered_labels = []

        for path, label in zip(data["paths"], data["labels"]):
            if label in class_set:
                filtered_paths.append(path)
                filtered_labels.append(label)

        return filtered_paths, filtered_labels

    def _load_and_decode(self, path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Load and decode a single image from disk."""
        raw = tf.io.read_file(path)
        image = tf.io.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.uint8)
        return image, label

    def _build_dataset(
        self,
        paths: List[str],
        labels: List[int],
        augment: bool = False,
        shuffle: bool = False,
        augmentation_config: Optional[dict] = None,
    ) -> tf.data.Dataset:
        """
        Build a tf.data pipeline from paths and labels.
        
        Pipeline: paths/labels → load_decode → normalize → [augment] → batch → prefetch
        """
        # Create dataset from paths and labels
        path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        if shuffle:
            path_ds = path_ds.shuffle(
                buffer_size=min(len(paths), 10000),
                seed=self.seed,
                reshuffle_each_iteration=True,
            )

        # Load and decode images
        dataset = path_ds.map(
            self._load_and_decode,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Normalize
        dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)

        # Augment (training only)
        if augment:
            aug_config = augmentation_config or {}
            aug_fn = create_augmentation_fn(
                image_size=self.image_size,
                seed=self.seed,
                **aug_config,
            )
            dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_task_train_dataset(
        self,
        task_id: int,
        augment: bool = True,
        augmentation_config: Optional[dict] = None,
    ) -> tf.data.Dataset:
        """
        Get the training dataset for a specific task.
        
        Args:
            task_id: 1-indexed task ID.
            augment: Whether to apply training augmentation.
            augmentation_config: Optional override for augmentation parameters.
        
        Returns:
            Batched, shuffled, augmented tf.data.Dataset.
        """
        task = self.task_generator.get_task(task_id)
        paths, labels = self._filter_by_classes(self.train_data, task.class_indices)
        
        print(f"[Loader] Task {task_id} train: {len(paths)} images, "
              f"classes {task.class_indices[:3]}...{task.class_indices[-1]}")

        return self._build_dataset(
            paths, labels,
            augment=augment,
            shuffle=True,
            augmentation_config=augmentation_config,
        )

    def get_task_val_dataset(self, task_id: int) -> tf.data.Dataset:
        """
        Get the validation dataset for a specific task.
        
        No augmentation, no shuffling.
        """
        task = self.task_generator.get_task(task_id)
        paths, labels = self._filter_by_classes(self.val_data, task.class_indices)
        
        print(f"[Loader] Task {task_id} val: {len(paths)} images")

        return self._build_dataset(paths, labels, augment=False, shuffle=False)

    def get_cumulative_val_dataset(self, up_to_task: int) -> tf.data.Dataset:
        """
        Get validation dataset for ALL classes from Task 1 through `up_to_task`.
        
        Used for measuring forgetting — evaluate on all previously learned classes.
        """
        all_classes = self.task_generator.get_classes_up_to_task(up_to_task)
        paths, labels = self._filter_by_classes(self.val_data, all_classes)
        
        print(f"[Loader] Cumulative val (tasks 1-{up_to_task}): "
              f"{len(paths)} images, {len(all_classes)} classes")

        return self._build_dataset(paths, labels, augment=False, shuffle=False)

    def get_all_val_dataset(self) -> tf.data.Dataset:
        """Get validation dataset for ALL 200 classes."""
        return self.get_cumulative_val_dataset(self.task_generator.num_tasks)

    def get_task_image_count(self, task_id: int, split: str = "train") -> int:
        """Get the number of images in a task."""
        task = self.task_generator.get_task(task_id)
        data = self.train_data if split == "train" else self.val_data
        paths, _ = self._filter_by_classes(data, task.class_indices)
        return len(paths)

    def benchmark_throughput(self, task_id: int = 1, num_batches: int = 50) -> float:
        """
        Measure data pipeline throughput (images/second).
        
        Per spec §11: GPU should not spend most time waiting for data.
        """
        import time
        
        dataset = self.get_task_train_dataset(task_id, augment=True)
        
        # Warm up
        for batch in dataset.take(5):
            pass

        start = time.perf_counter()
        count = 0
        for batch_images, batch_labels in dataset.take(num_batches):
            count += batch_images.shape[0]
        elapsed = time.perf_counter() - start

        throughput = count / elapsed
        print(f"[Benchmark] Throughput: {throughput:.0f} images/sec "
              f"({count} images in {elapsed:.2f}s)")
        return throughput


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_default_config

    config = get_default_config()

    try:
        task_gen = TaskGenerator(
            data_dir=config.dataset.data_dir,
            num_tasks=config.dataset.num_tasks,
            classes_per_task=config.dataset.classes_per_task,
            seed=config.dataset.seed,
        )

        loader = TinyImageNetLoader(
            data_dir=config.dataset.data_dir,
            task_generator=task_gen,
            batch_size=config.training.batch_size,
            image_size=config.dataset.image_size,
            seed=config.dataset.seed,
        )

        # Test loading Task 1
        train_ds = loader.get_task_train_dataset(1)
        for images, labels in train_ds.take(1):
            print(f"\n[Test] Batch shape: images={images.shape}, labels={labels.shape}")
            print(f"[Test] Image range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
            print(f"[Test] Labels sample: {labels.numpy()[:10]}")

        # Benchmark
        loader.benchmark_throughput(task_id=1)

    except FileNotFoundError as e:
        print(f"[Loader] Dataset not found: {e}")
        print("[Loader] Run `python data/download.py` first.")
