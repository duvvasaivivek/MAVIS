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
    """
    def __init__(self, in_features: int, out_features: int, scale: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("task_scale", torch.ones(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2-normalize features along dim=1, weights along dim=1
        norm_x = F.normalize(x, p=2, dim=1)
        norm_w = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(norm_x, norm_w)
        return self.scale * (logits * self.task_scale)


class IncrementalClassifier(nn.Module):
    """
    A balanced classification head mapping feature vectors to class logits
    using CosineLinear with Weight Aligning (WA).
    """
    def __init__(self, feature_dim: int, total_classes: int = 200, scale: float = 16.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.total_classes = total_classes
        self.fc = CosineLinear(feature_dim, total_classes, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @torch.no_grad()
    def align_weights(self, num_old_classes: int, num_new_classes: int, prototypes: torch.Tensor = None):
        """
        Weight Aligning (WA - CVPR 2020 & Alignment Calibration):
        Calibrates the effective logits of newly trained classes with historical classes,
        removing classifier recency bias so old tasks don't get drowned out.
        """
        if num_old_classes <= 0 or num_new_classes <= 0:
            return
        
        old_w = self.fc.weight[:num_old_classes]
        new_w = self.fc.weight[num_old_classes:num_old_classes + num_new_classes]

        # 1. Classical WA norm ratio (Zhao et al., CVPR 2020)
        old_norm = torch.norm(old_w, p=2, dim=1).mean()
        new_norm = torch.norm(new_w, p=2, dim=1).mean()
        gamma = (old_norm / new_norm).item() if new_norm > 0 else 1.0

        # 2. Alignment calibration if prototypes are available
        if prototypes is not None and len(prototypes) >= (num_old_classes + num_new_classes):
            norm_old_w = F.normalize(old_w, p=2, dim=1)
            norm_new_w = F.normalize(new_w, p=2, dim=1)
            norm_old_p = F.normalize(prototypes[:num_old_classes].to(self.fc.weight.device), p=2, dim=1)
            norm_new_p = F.normalize(prototypes[num_old_classes:num_old_classes + num_new_classes].to(self.fc.weight.device), p=2, dim=1)

            old_align = (norm_old_w * norm_old_p).sum(dim=1).mean().item()
            new_align = (norm_new_w * norm_new_p).sum(dim=1).mean().item()

            if new_align > 0 and old_align > 0:
                gamma_align = old_align / new_align
                gamma = min(gamma, gamma_align)

        gamma = max(0.2, min(1.0, float(gamma)))
        self.fc.task_scale[num_old_classes:num_old_classes + num_new_classes] = gamma
        print(f"  [WeightAlign] Set gamma={gamma:.4f} for classes [{num_old_classes}..{num_old_classes + num_new_classes - 1}]")
