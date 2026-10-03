"""
Tiny ImageNet Downloader
========================
Downloads and extracts the Tiny ImageNet dataset (200 classes, 64×64×3).
Also provides CIFAR-100 download for development/debugging.
"""

import os
import zipfile
import shutil
import time
import urllib.request
from typing import Optional

# Tiny ImageNet URL
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_file(url: str, dest_path: str, desc: Optional[str] = None) -> None:
    """Download a file with progress reporting."""
    if os.path.exists(dest_path):
        print(f"[Download] Already exists: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    desc = desc or os.path.basename(dest_path)
    print(f"[Download] Downloading {desc} from {url}...")

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  Progress: {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=_report)
    print(f"\n[Download] Saved to {dest_path}")


def extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract a zip file."""
    print(f"[Extract] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    print(f"[Extract] Done → {extract_dir}")


def prepare_tiny_imagenet_val(data_dir: str) -> None:
    """
    Reorganize the Tiny ImageNet validation set into class subdirectories.
    
    The default structure has all val images in one folder with a separate
    annotations file. We reorganize to match the training structure:
        val/<class_id>/images/<image_file>
    """
    val_dir = os.path.join(data_dir, "val")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(annotations_file):
        print("[Prepare] val_annotations.txt not found — validation may already be organized.")
        return

    print("[Prepare] Reorganizing validation set into class subdirectories...")

    # Read annotations: filename -> class_id
    with open(annotations_file, "r") as f:
        annotations = {}
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                annotations[parts[0]] = parts[1]

    val_images_dir = os.path.join(val_dir, "images")
    moved_count = 0

    for filename, class_id in annotations.items():
        src = os.path.join(val_images_dir, filename)
        if not os.path.exists(src):
            continue

        dst_dir = os.path.join(val_dir, class_id, "images")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename)
        
        # Windows sometimes holds locks on newly extracted files. Retry with backoff.
        for _ in range(5):
            try:
                shutil.move(src, dst)
                break
            except PermissionError:
                time.sleep(0.1)
        else:
            print(f"[Prepare] Warning: Could not move {filename} due to locks.")
            continue
            
        moved_count += 1

    # Clean up the old flat images directory if empty
    if os.path.exists(val_images_dir) and len(os.listdir(val_images_dir)) == 0:
        os.rmdir(val_images_dir)

    print(f"[Prepare] Moved {moved_count} validation images into {len(set(annotations.values()))} class dirs.")


def download_tiny_imagenet(data_dir: str) -> str:
    """
    Download and prepare Tiny ImageNet.
    
    Returns:
        Path to the tiny-imagenet-200 directory.
    """
    dataset_dir = os.path.join(data_dir, "tiny-imagenet-200")

    if os.path.exists(dataset_dir) and os.path.isdir(os.path.join(dataset_dir, "train")):
        print(f"[Download] Tiny ImageNet already exists at {dataset_dir}")
    else:
        zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
        download_file(TINY_IMAGENET_URL, zip_path, desc="Tiny ImageNet")
        extract_zip(zip_path, data_dir)

    # Reorganize validation set
    prepare_tiny_imagenet_val(dataset_dir)

    # Verify
    train_dir = os.path.join(dataset_dir, "train")
    if os.path.exists(train_dir):
        num_classes = len([d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))])
        print(f"[Verify] Training classes found: {num_classes}")
    
    return dataset_dir


def download_cifar100() -> str:
    """
    Download CIFAR-100 via TensorFlow/Keras (for development/debugging).
    
    Returns:
        String indicating CIFAR-100 is loaded via tf.keras.datasets.
    """
    print("[Download] CIFAR-100 will be loaded via tf.keras.datasets.cifar100")
    print("[Download] It downloads automatically on first use (~169 MB)")
    return "cifar100"


def verify_dataset(data_dir: str) -> dict:
    """
    Verify dataset integrity: image count, class count, dimensions.
    
    Returns:
        Dictionary with verification results.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    results = {
        "data_dir": data_dir,
        "train_classes": 0,
        "train_images": 0,
        "val_classes": 0,
        "val_images": 0,
        "sample_dimensions": None,
        "issues": [],
    }

    # Count training data
    if os.path.exists(train_dir):
        class_dirs = [d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        results["train_classes"] = len(class_dirs)
        for cls_dir in class_dirs:
            images_dir = os.path.join(train_dir, cls_dir, "images")
            if os.path.exists(images_dir):
                img_files = [f for f in os.listdir(images_dir) if f.endswith(".JPEG")]
                results["train_images"] += len(img_files)
    else:
        results["issues"].append("Training directory not found")

    # Count validation data
    if os.path.exists(val_dir):
        class_dirs = [d for d in os.listdir(val_dir) 
                      if os.path.isdir(os.path.join(val_dir, d))]
        results["val_classes"] = len(class_dirs)
        for cls_dir in class_dirs:
            images_dir = os.path.join(val_dir, cls_dir, "images")
            if os.path.exists(images_dir):
                img_files = [f for f in os.listdir(images_dir) if f.endswith(".JPEG")]
                results["val_images"] += len(img_files)

    # Check a sample image dimension
    if results["train_images"] > 0:
        try:
            from PIL import Image
            sample_cls = os.listdir(train_dir)[0]
            sample_img_dir = os.path.join(train_dir, sample_cls, "images")
            sample_img = os.listdir(sample_img_dir)[0]
            img = Image.open(os.path.join(sample_img_dir, sample_img))
            results["sample_dimensions"] = f"{img.size[0]}×{img.size[1]}, mode={img.mode}"
        except Exception as e:
            results["issues"].append(f"Could not read sample image: {e}")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATA_DIR

    dataset_dir = download_tiny_imagenet(DATA_DIR)
    info = verify_dataset(dataset_dir)

    print("\n" + "=" * 50)
    print("Dataset Verification Results")
    print("=" * 50)
    for k, v in info.items():
        print(f"  {k}: {v}")
