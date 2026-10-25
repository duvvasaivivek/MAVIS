"""
PyTorch DataLoader for Tiny ImageNet Continual Tasks
====================================================
"""

import os
import hashlib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from data.task_generator import TaskGenerator
from data.augmentation import get_train_transforms, get_eval_transforms

# Decoded-image disk cache (added 2026-10-25). Decoding one task's 10,000 JPEGs with PIL was
# measured at ~235 s per process; the review factorial launches one process per (cell, seed), so
# decoding dominated the run. The cache stores the already-RGB-converted uint8 pixels, so an item
# served from it is byte-identical to one decoded from the JPEG. Lives inside the (gitignored)
# dataset directory. Delete the folder to force a re-decode.
CACHE_DIRNAME = "_decoded_cache"


class TinyImageNetTaskDataset(Dataset):
    """A PyTorch Dataset for a specific subset of Tiny ImageNet classes."""
    def __init__(self, data_dir: str, class_list: list, split: str = 'train', transform=None,
                 use_disk_cache: bool = True):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.samples = []
        
        # Build dataset mapping
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} not found.")
            
        # Tiny ImageNet classes are stored in WNID folders (n0xxxxxx)
        # Class list contains the names of the folders we want for this task
        for class_idx, class_name in enumerate(class_list):
            class_dir = os.path.join(split_dir, class_name)
            img_dir = os.path.join(class_dir, 'images')
                
            if not os.path.exists(img_dir):
                continue
                
            for img_name in os.listdir(img_dir):
                if img_name.endswith('.JPEG'):
                    self.samples.append((os.path.join(img_dir, img_name), class_idx))

        # RAM Preloading: 10,000 64x64 images take only ~120 MB RAM.
        # Preloading completely eliminates disk I/O bottlenecks and Python 3.14 IPC crashes.
        # Pixels are held as one uint8 array [N, 64, 64, 3] and handed to the transform as a PIL
        # image per item, which is what the previous list-of-PIL-images version did.
        self.pixels = None
        self.cached_images = []
        if use_disk_cache:
            self.pixels = self._load_or_build_cache(class_list)
        if self.pixels is None:
            for path, _ in self.samples:
                with open(path, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                self.cached_images.append(img)

    def _cache_path(self, class_list) -> str:
        key = hashlib.md5(("|".join(class_list)).encode("utf-8")).hexdigest()[:12]
        cache_dir = os.path.join(self.data_dir, CACHE_DIRNAME)
        return os.path.join(cache_dir, f"{self.split}_{key}_{len(self.samples)}.npy")

    def _load_or_build_cache(self, class_list):
        """Return a uint8 array [N, H, W, 3], or None if the images are not uniformly sized."""
        path = self._cache_path(class_list)
        if os.path.exists(path):
            try:
                arr = np.load(path, mmap_mode="r")
                if arr.shape[0] == len(self.samples) and arr.ndim == 4 and arr.shape[3] == 3:
                    return arr
            except Exception:
                pass  # fall through and rebuild

        decoded = []
        shape = None
        for img_path, _ in self.samples:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            a = np.asarray(img, dtype=np.uint8)
            if shape is None:
                shape = a.shape
            elif a.shape != shape:
                # Mixed resolutions: fall back to the PIL list so no resizing is introduced here.
                self.cached_images = [Image.fromarray(x) for x in decoded]
                self.cached_images.append(img)
                for p2, _ in self.samples[len(self.cached_images):]:
                    with open(p2, 'rb') as f2:
                        self.cached_images.append(Image.open(f2).convert('RGB'))
                return None
            decoded.append(a)

        arr = np.stack(decoded, axis=0) if decoded else np.zeros((0, 64, 64, 3), dtype=np.uint8)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + f".tmp{os.getpid()}"
            # Write through an open handle: np.save() appends ".npy" to a *path* that lacks it,
            # which would leave the temp file under a different name than os.replace() expects.
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[Data] Could not write decode cache {path}: {e}", flush=True)
        return arr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target = self.samples[idx][1]
        if self.pixels is not None:
            img = Image.fromarray(np.asarray(self.pixels[idx]))
        elif self.cached_images:
            img = self.cached_images[idx]
        else:
            path, target = self.samples[idx]
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class TinyImageNetLoader:
    """Manages PyTorch DataLoaders for Continual Learning tasks."""
    def __init__(self, data_dir: str, task_generator: TaskGenerator, batch_size: int, num_workers: int = 0, aug_config=None):
        self.data_dir = data_dir
        self.task_generator = task_generator
        self.batch_size = batch_size
        self.num_workers = 0  # CRITICAL: num_workers=0 completely eliminates Python 3.14 multiprocessing / sem_unlink bugs
        self.train_transform = get_train_transforms(aug_config) if aug_config else get_eval_transforms()
        self.eval_transform = get_eval_transforms()
        # Decoded datasets are cached per (task, split). The MAVIS trainer re-requests every past
        # task's training loader after each task (prototype re-alignment), which previously re-decoded
        # 10,000 JPEGs each time. Caching changes nothing about the data; transforms still run per item.
        self._dataset_cache = {}

    def _get_dataset(self, task_id: int, split: str) -> TinyImageNetTaskDataset:
        key = (task_id, split)
        if key not in self._dataset_cache:
            class_list = self.task_generator.get_task(task_id + 1).class_names
            transform = self.train_transform if split == 'train' else self.eval_transform
            self._dataset_cache[key] = TinyImageNetTaskDataset(self.data_dir, class_list, split=split, transform=transform)
        return self._dataset_cache[key]

    def get_task_train_dataloader(self, task_id: int) -> DataLoader:
        dataset = self._get_dataset(task_id, 'train')
        # In-process DataLoader is blazing fast with RAM caching and immune to POSIX semaphore leaks
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )

    def get_task_val_dataloader(self, task_id: int) -> DataLoader:
        dataset = self._get_dataset(task_id, 'val')
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )

    def get_cumulative_val_dataloader(self, up_to_task: int) -> DataLoader:
        cumulative_classes = []
        for t in range(up_to_task + 1):
            cumulative_classes.extend(self.task_generator.get_task(t + 1).class_names)
        dataset = TinyImageNetTaskDataset(self.data_dir, cumulative_classes, split='val', transform=self.eval_transform)
        # num_workers=0 for validation avoids multiprocessing IPC semaphore leaks in Python 3.14
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
