"""
Decode-cache equivalence test (added 2026-10-25)
===============================================
The factorial study launches one process per (cell, seed), and each process re-decoded every
task's 10,000 JPEGs with PIL (~235 s measured). A uint8 disk cache removes that cost, but only if
an item served from the cache is identical to one decoded from the JPEG.

This test would fail if the cache reordered samples, dropped the RGB conversion, resized anything,
or mismatched labels. It compares the cached and uncached datasets item by item with transforms
switched off (so no randomness is involved), then checks that an identical transform pipeline under
an identical seed yields identical tensors from both paths.

Run:  python tests/test_decode_cache.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TRITON_DISABLE"] = "1"

import time
import numpy as np
import torch

from config import get_default_config
from data.task_generator import TaskGenerator
from data.tiny_imagenet import TinyImageNetTaskDataset
from data.augmentation import get_train_transforms, get_eval_transforms


def main():
    cfg = get_default_config()
    tg = TaskGenerator(cfg.dataset.data_dir, cfg.dataset.num_tasks,
                       cfg.dataset.classes_per_task, cfg.dataset.seed)
    names = tg.get_task(1).class_names

    # 1. Raw pixel equivalence, no transform, on the validation split (1,000 images).
    t0 = time.time()
    plain = TinyImageNetTaskDataset(cfg.dataset.data_dir, names, 'val', transform=None,
                                    use_disk_cache=False)
    t_plain = time.time() - t0
    t0 = time.time()
    cached = TinyImageNetTaskDataset(cfg.dataset.data_dir, names, 'val', transform=None,
                                     use_disk_cache=True)
    t_cached = time.time() - t0

    assert len(plain) == len(cached), (len(plain), len(cached))
    assert [p for p, _ in plain.samples] == [p for p, _ in cached.samples], "sample order differs"

    mismatch = 0
    for i in range(len(plain)):
        a_img, a_lab = plain[i]
        b_img, b_lab = cached[i]
        assert a_lab == b_lab, f"label mismatch at {i}: {a_lab} vs {b_lab}"
        a = np.asarray(a_img, dtype=np.uint8)
        b = np.asarray(b_img, dtype=np.uint8)
        assert a.shape == b.shape, f"shape mismatch at {i}: {a.shape} vs {b.shape}"
        if not np.array_equal(a, b):
            mismatch += 1
    assert mismatch == 0, f"{mismatch} of {len(plain)} images differ between cached and decoded"
    print(f"[OK] 1. {len(plain)} val images identical through the cache; labels identical; order identical")
    print(f"       build/decode {t_plain:.1f}s uncached vs {t_cached:.1f}s cached")

    # 2. Same transform + same seed -> same tensor from both paths (eval transform, deterministic).
    ev = get_eval_transforms()
    p2 = TinyImageNetTaskDataset(cfg.dataset.data_dir, names, 'val', transform=ev, use_disk_cache=False)
    c2 = TinyImageNetTaskDataset(cfg.dataset.data_dir, names, 'val', transform=ev, use_disk_cache=True)
    for i in (0, 1, 7, 100, len(p2) - 1):
        xa, ya = p2[i]
        xb, yb = c2[i]
        assert ya == yb and torch.equal(xa, xb), f"eval tensor differs at {i}"
    print("[OK] 2. eval-transform tensors identical from both paths")

    # 3. Random train transform: identical under a fixed seed, because the only randomness is the
    #    transform's own, not the pixel source.
    tr = get_train_transforms(cfg.augmentation)
    p3 = TinyImageNetTaskDataset(cfg.dataset.data_dir, names, 'val', transform=tr, use_disk_cache=False)
    c3 = TinyImageNetTaskDataset(cfg.dataset.data_dir, names, 'val', transform=tr, use_disk_cache=True)
    for i in (0, 3, 42):
        torch.manual_seed(1234)
        xa, _ = p3[i]
        torch.manual_seed(1234)
        xb, _ = c3[i]
        assert torch.equal(xa, xb), f"train-transform tensor differs at {i} under a fixed seed"
    print("[OK] 3. train-transform tensors identical under a fixed seed")

    # 4. Cache file exists and is the expected dtype/shape.
    path = cached._cache_path(names)
    assert os.path.exists(path), path
    arr = np.load(path, mmap_mode="r")
    assert arr.dtype == np.uint8 and arr.shape == (len(cached), 64, 64, 3), (arr.dtype, arr.shape)
    size_mb = os.path.getsize(path) / 1e6
    print(f"[OK] 4. cache file is uint8 {arr.shape}, {size_mb:.1f} MB")

    print("\nDecode-cache equivalence verified.")


if __name__ == "__main__":
    main()
