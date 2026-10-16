"""
Neural Memory Bank (Phase 4 & 5)
================================
Stores frozen multi-prototype features for each class in each completed task.
Uses K-Means clustering (K=3) to preserve multi-modal visual class distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kmeans_prototypes(features: torch.Tensor, k: int = 3, num_iters: int = 10) -> torch.Tensor:
    """
    Computes k prototype cluster centroids for a class feature set using pure PyTorch.
    Returns normalized centroids [k, D].
    """
    N, D = features.size()
    if N <= k:
        norm_f = F.normalize(features, p=2, dim=1)
        if N < k:
            repeat_factor = (k + N - 1) // N
            norm_f = norm_f.repeat(repeat_factor, 1)[:k]
        return norm_f

    # K-Means++ style spread out initialization
    indices = torch.linspace(0, N - 1, steps=k).long()
    centroids = features[indices].clone()

    for _ in range(num_iters):
        dists = torch.cdist(features, centroids)
        cluster_assignments = dists.argmin(dim=1)

        new_centroids = []
        for i in range(k):
            mask = (cluster_assignments == i)
            if mask.any():
                new_centroids.append(features[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[i])
        centroids = torch.stack(new_centroids, dim=0)

    return F.normalize(centroids, p=2, dim=1)


class MemoryBank(nn.Module):
    """
    Multi-Prototype Memory Bank storing K prototypes per class per task.
    Shape: [num_tasks, classes_per_task, prototypes_per_class, feature_dim]
    """
    def __init__(self, feature_dim: int, num_tasks: int, classes_per_task: int, prototypes_per_class: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.prototypes_per_class = prototypes_per_class

        # Registered buffers: saved with model state_dict but NOT trained by optimizer
        self.register_buffer("memory", torch.zeros(num_tasks, classes_per_task, prototypes_per_class, feature_dim))
        self.register_buffer("is_populated", torch.zeros(num_tasks, dtype=torch.bool))
        self.register_buffer("class_variances", torch.zeros(num_tasks, classes_per_task, feature_dim))

        # Online running prototypes for the current active task during training
        self.register_buffer("running_prototypes", torch.zeros(classes_per_task, feature_dim))
        self.register_buffer("running_counts", torch.zeros(classes_per_task))

    @torch.no_grad()
    def update_running_prototypes(self, features: torch.Tensor, targets: torch.Tensor):
        """
        Online update of current task's prototypes during training so that
        Memory Attention has valid targets to attend to from Epoch 1.
        """
        features_d = features.detach().float()
        for c in range(self.classes_per_task):
            mask = (targets == c)
            if mask.any():
                batch_mean = features_d[mask].mean(dim=0)
                if self.running_counts[c] == 0:
                    self.running_prototypes[c] = batch_mean
                else:
                    self.running_prototypes[c] = 0.92 * self.running_prototypes[c] + 0.08 * batch_mean
                self.running_counts[c] += 1

    def reset_running_prototypes(self):
        """Resets running buffers at task transitions."""
        self.running_prototypes.zero_()
        self.running_counts.zero_()

    def get_current_task_prototypes(self):
        """
        Returns normalized running prototypes for the active task [classes_per_task, D].
        """
        valid_running = []
        for c in range(self.classes_per_task):
            if self.running_counts[c] > 0:
                valid_running.append(F.normalize(self.running_prototypes[c], p=2, dim=0))
            else:
                valid_running.append(torch.zeros(self.feature_dim, device=self.memory.device))
        return torch.stack(valid_running, dim=0).to(self.memory.device)

    @torch.no_grad()
    def update_memory(self, task_id: int, train_loader, encoder, device):
        """
        Compute and store K feature prototypes and class variances for each class in this task.
        """
        encoder.eval()
        class_features = [[] for _ in range(self.classes_per_task)]

        for inputs, targets in train_loader:
            inputs = inputs.to(device)

            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                features = encoder(inputs)

            features_cpu = features.float().cpu()
            for i in range(len(targets)):
                c = targets[i].item()
                class_features[c].append(features_cpu[i])

        # Compute K prototypes per class using K-Means clustering + diagonal feature variances
        for c in range(self.classes_per_task):
            if len(class_features[c]) > 0:
                c_feats = torch.stack(class_features[c], dim=0)
                protos_k = kmeans_prototypes(c_feats, k=self.prototypes_per_class, num_iters=10)
                self.memory[task_id, c] = protos_k.to(self.memory.device)
                var_c = c_feats.var(dim=0, unbiased=False) + 1e-4
                self.class_variances[task_id, c] = var_c.to(self.memory.device)

        self.is_populated[task_id] = True
        self.reset_running_prototypes()
        encoder.train()

    def get_past_memories(self, current_task_id: int):
        """
        Returns prototypes from tasks 0..current_task_id-1 ONLY, flattened to [N * K, D].
        """
        if current_task_id <= 0:
            return None

        valid_indices = []
        for t in range(current_task_id):
            if self.is_populated[t]:
                valid_indices.append(t)

        if len(valid_indices) == 0:
            return None

        indices = torch.tensor(valid_indices, device=self.memory.device)
        return self.memory[indices].reshape(-1, self.feature_dim)

    def get_past_prototypes_and_labels(self, current_task_id: int):
        """
        Returns all stored prototypes, class labels, and diagonal variances
        for tasks 0..current_task_id-1 (Gaussian Feature Covariance Replay).
        Returns (prototypes [N_proto, D], labels [N_proto], variances [N_proto, D]).
        """
        if current_task_id <= 0:
            return None, None, None

        valid_indices = []
        for t in range(current_task_id):
            if self.is_populated[t]:
                valid_indices.append(t)

        if len(valid_indices) == 0:
            return None, None, None

        indices = torch.tensor(valid_indices, device=self.memory.device)
        selected_mem = self.memory[indices]  # [len(valid_indices), classes_per_task, K, D]
        num_past_classes = len(valid_indices) * self.classes_per_task
        protos = selected_mem.reshape(num_past_classes * self.prototypes_per_class, self.feature_dim)

        task_labels = []
        task_vars = []
        for t in valid_indices:
            t_labels = torch.arange(t * self.classes_per_task, (t + 1) * self.classes_per_task, device=self.memory.device).repeat_interleave(self.prototypes_per_class)
            task_labels.append(t_labels)
            t_vars = self.class_variances[t].repeat_interleave(self.prototypes_per_class, dim=0)
            task_vars.append(t_vars)
        labels = torch.cat(task_labels, dim=0)
        variances = torch.cat(task_vars, dim=0)

        return protos, labels, variances

    def get_training_memories(self, current_task_id: int):
        """
        Returns all memories available during training:
        1. Frozen locked prototypes from all completed past tasks
        2. Normalized running prototypes from the current active task
        """
        past = self.get_past_memories(current_task_id)

        if self.running_counts.sum() > 0:
            curr = self.get_current_task_prototypes()
            if past is not None:
                return torch.cat([past, curr], dim=0)
            else:
                return curr
        else:
            return past

    def get_all_populated_memories(self):
        """
        Returns ALL populated prototypes flattened to [num_seen_classes * K, D].
        """
        valid_tasks = self.is_populated.nonzero(as_tuple=True)[0]
        if len(valid_tasks) == 0:
            return None
        return self.memory[valid_tasks].reshape(-1, self.feature_dim)

    def get_mean_class_prototypes(self, up_to_task_id: int = None):
        """
        Returns the mean prototype for each class [num_seen_classes, feature_dim].
        Useful for alignment calibration and global distance monitoring.
        """
        valid_tasks = self.is_populated.nonzero(as_tuple=True)[0]
        if len(valid_tasks) == 0:
            return None
        if up_to_task_id is not None:
            valid_tasks = valid_tasks[valid_tasks <= up_to_task_id]
            if len(valid_tasks) == 0:
                return None
        mean_protos = self.memory[valid_tasks].mean(dim=2)
        return mean_protos.reshape(-1, self.feature_dim)

    def compute_ncm_scores(self, features: torch.Tensor, up_to_task_id: int = None):
        """
        Multi-Prototype Nearest Class Mean classification.
        Computes cosine similarity between query features and all populated prototypes,
        taking the max similarity over the K prototypes for each class.

        Args:
            features: [B, D] query features
            up_to_task_id: optional int limit
        Returns:
            class_scores: [B, num_seen_classes]
        """
        valid_tasks = self.is_populated.nonzero(as_tuple=True)[0]
        if len(valid_tasks) == 0:
            return None

        if up_to_task_id is not None:
            valid_tasks = valid_tasks[valid_tasks <= up_to_task_id]
            if len(valid_tasks) == 0:
                return None

        num_tasks = len(valid_tasks)
        num_classes = num_tasks * self.classes_per_task
        protos = self.memory[valid_tasks].reshape(num_classes * self.prototypes_per_class, self.feature_dim)
        protos = F.normalize(protos, p=2, dim=1)  # Ensure unit-norm after re-encoding

        norm_f = F.normalize(features, p=2, dim=1)
        sim = F.linear(norm_f, protos)
        sim = sim.view(-1, num_classes, self.prototypes_per_class)
        class_scores, _ = sim.max(dim=2)
        return class_scores
