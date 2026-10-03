"""
Classifier Head (PyTorch)
=========================
Dynamic or static classification head mapping features to class logits.
"""

import torch
import torch.nn as nn

class IncrementalClassifier(nn.Module):
    """
    A classification head that maps feature vectors to class logits.
    """
    def __init__(self, feature_dim: int, total_classes: int = 200):
        super().__init__()
        self.feature_dim = feature_dim
        self.total_classes = total_classes
        self.fc = nn.Linear(feature_dim, total_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
