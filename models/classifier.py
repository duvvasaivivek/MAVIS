"""
Classifier Head (PyTorch)
=========================
Cosine Linear classification head mapping features to normalized class logits.
Eliminates inter-task logit magnitude imbalance and protects older classes
from AdamW decoupled weight decay disparity in Class-Incremental Learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLinear(nn.Module):
    """
    Cosine Linear classification layer with temperature scaling and Weight Aligning.
    Normalizes both weights and features to unit sphere:
        logits = scale * (x / ||x||) @ (w / ||w||)^T * task_scale
    Supports additive cosine margin for inter-class angular separation.
    """
    def __init__(self, in_features: int, out_features: int, scale: float = 24.0, margin: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("task_scale", torch.ones(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        # L2-normalize features along dim=1, weights along dim=1
        norm_x = F.normalize(x, p=2, dim=1)
        norm_w = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(norm_x, norm_w)
        if self.training and self.margin > 0.0 and targets is not None:
            # Additive cosine margin: subtract margin from ground-truth class cosine similarity
            one_hot = F.one_hot(targets, self.out_features).float()
            logits = logits - one_hot * self.margin
        return self.scale * (logits * self.task_scale)


class IncrementalClassifier(nn.Module):
    """
    A balanced classification head mapping feature vectors to class logits
    using CosineLinear with Weight Aligning (WA) and Cosine Margin.
    """
    def __init__(self, feature_dim: int, total_classes: int = 200, scale: float = 24.0, margin: float = 0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.total_classes = total_classes
        self.fc = CosineLinear(feature_dim, total_classes, scale=scale, margin=margin)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        return self.fc(x, targets=targets)

    @torch.no_grad()
    def align_weights(self, num_old_classes: int, num_new_classes: int):
        """
        Measure 1: Equalize classifier head temperature scales.
        Ensures all seen classes operate on an identical logit scale (1.0),
        eliminating cross-task softmax temperature suppression.
        """
        total_seen = num_old_classes + num_new_classes
        self.fc.task_scale[:total_seen] = 1.0
        print(f"  [Measure 1: WeightAlign] Equalized temperature scale to 1.0 across all {total_seen} seen classes.", flush=True)
