"""
Neural Memory Bank (Phase 4)
==============================
Stores frozen prototype features for each class in each completed task.

DATA LEAKAGE PREVENTION:
  - Prototypes are computed from TRAINING data only (never validation data).
  - Prototypes are computed AFTER training on that task completes (post-hoc).
  - Prototypes use eval-mode encoder with no_grad (no augmentation influence).
  - Memory is a registered buffer (frozen by design, never touched by backprop).
  - get_past_memories(task_id) returns ONLY tasks 0..task_id-1 (never the current task).

OVERFITTING PREVENTION:
  - Prototypes are L2-normalized to a unit sphere (prevents magnitude explosion).
  - Each prototype is the MEAN of all samples for that class (smooths out noise).
  - Memory is never trained — it is purely a frozen lookup table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """
    Fixed-size memory storing one prototype per class per task.
    Shape: [num_tasks, classes_per_task, feature_dim]
    """
    def __init__(self, feature_dim: int, num_tasks: int, classes_per_task: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task

        # Registered buffers: saved with model state_dict but NOT trained by optimizer
        self.register_buffer("memory", torch.zeros(num_tasks, classes_per_task, feature_dim))
        self.register_buffer("is_populated", torch.zeros(num_tasks, dtype=torch.bool))

    @torch.no_grad()
    def update_memory(self, task_id: int, train_loader, encoder, device):
        """
        Compute and store the mean feature prototype for each class in this task.

        CRITICAL: This uses the TRAINING dataloader (not validation) and runs
        the encoder in eval mode with no_grad to get clean, deterministic features.
        The eval-mode encoder uses NO augmentation because the train_loader's
        transforms include augmentation, but we process features through
        the deterministic forward pass of the encoder (Dropout disabled in eval mode,
        GroupNorm uses learned parameters deterministically).
        """
        encoder.eval()
        class_sums = torch.zeros(self.classes_per_task, self.feature_dim, device='cpu')
        class_counts = torch.zeros(self.classes_per_task, device='cpu')

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            # targets are already 0-indexed within the task (0..classes_per_task-1)
            # because TinyImageNetTaskDataset uses class_idx as the target

            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                features = encoder(inputs)

            # Accumulate on CPU to avoid GPU memory pressure
            features_cpu = features.float().cpu()
            for i in range(len(targets)):
                c = targets[i].item()
                class_sums[c] += features_cpu[i]
                class_counts[c] += 1

        # Compute mean prototype per class and L2-normalize
        for c in range(self.classes_per_task):
            if class_counts[c] > 0:
                proto = class_sums[c] / class_counts[c]
                proto = F.normalize(proto, p=2, dim=0)
                self.memory[task_id, c] = proto

        self.is_populated[task_id] = True
        encoder.train()

    def get_past_memories(self, current_task_id: int):
        """
        Returns prototypes from tasks 0..current_task_id-1 ONLY.

        DATA LEAKAGE PREVENTION:
          - During training on task T, the model must NOT see task T's own
            prototypes (they are being learned right now).
          - It may only retrieve memories from tasks that are fully completed.

        Returns:
            Tensor of shape [N, feature_dim] where N = populated_past_tasks * classes_per_task
            or None if no past tasks exist.
        """
        if current_task_id <= 0:
            return None

        # Collect only tasks that are (a) populated AND (b) strictly before current_task_id
        valid_indices = []
        for t in range(current_task_id):
            if self.is_populated[t]:
                valid_indices.append(t)

        if len(valid_indices) == 0:
            return None

        indices = torch.tensor(valid_indices, device=self.memory.device)
        return self.memory[indices].reshape(-1, self.feature_dim)

    def get_all_populated_memories(self):
        """
        Returns ALL populated prototypes. Used ONLY during evaluation (never training).
        """
        valid_tasks = self.is_populated.nonzero(as_tuple=True)[0]
        if len(valid_tasks) == 0:
            return None
        return self.memory[valid_tasks].reshape(-1, self.feature_dim)
