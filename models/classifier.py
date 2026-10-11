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
    Cosine Linear classification layer with temperature scaling.
    Normalizes both weights and features to unit sphere:
        logits = scale * (x / ||x||) @ (w / ||w||)^T
    """
    def __init__(self, in_features: int, out_features: int, scale: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2-normalize features along dim=1, weights along dim=1
        norm_x = F.normalize(x, p=2, dim=1)
        norm_w = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(norm_x, norm_w)


class IncrementalClassifier(nn.Module):
    """
    A balanced classification head mapping feature vectors to class logits
    using CosineLinear.
    """
    def __init__(self, feature_dim: int, total_classes: int = 200, scale: float = 16.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.total_classes = total_classes
        self.fc = CosineLinear(feature_dim, total_classes, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
