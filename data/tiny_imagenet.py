"""
PyTorch DataLoader for Tiny ImageNet Continual Tasks
====================================================
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from data.task_generator import TaskGenerator
from data.augmentation import get_train_transforms, get_eval_transforms

class TinyImageNetTaskDataset(Dataset):
    """A PyTorch Dataset for a specific subset of Tiny ImageNet classes."""
    def __init__(self, data_dir: str, class_list: list, split: str = 'train', transform=None):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # Open as RGB
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target

class TinyImageNetLoader:
    """Manages PyTorch DataLoaders for Continual Learning tasks."""
    def __init__(self, data_dir: str, task_generator: TaskGenerator, batch_size: int, num_workers: int = 4, aug_config=None):
        self.data_dir = data_dir
        self.task_generator = task_generator
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = get_train_transforms(aug_config) if aug_config else get_eval_transforms()
        self.eval_transform = get_eval_transforms()

    def get_task_train_dataloader(self, task_id: int) -> DataLoader:
        class_list = self.task_generator.get_task(task_id + 1).class_names
        dataset = TinyImageNetTaskDataset(self.data_dir, class_list, split='train', transform=self.train_transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def get_task_val_dataloader(self, task_id: int) -> DataLoader:
        class_list = self.task_generator.get_task(task_id + 1).class_names
        dataset = TinyImageNetTaskDataset(self.data_dir, class_list, split='val', transform=self.eval_transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def get_cumulative_val_dataloader(self, up_to_task: int) -> DataLoader:
        cumulative_classes = []
        for t in range(up_to_task + 1):
            cumulative_classes.extend(self.task_generator.get_task(t + 1).class_names)
        dataset = TinyImageNetTaskDataset(self.data_dir, cumulative_classes, split='val', transform=self.eval_transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
