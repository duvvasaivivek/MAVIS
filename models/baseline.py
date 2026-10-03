"""
Baseline CNN Model (PyTorch)
============================
Combines the Encoder and Classifier into an end-to-end model.
"""

import torch
import torch.nn as nn

from models.encoder import CNNEncoder
from models.classifier import IncrementalClassifier

class BaselineCNN(nn.Module):
    """
    Standard CNN Model for Class-Incremental Learning.
    
    Forward pass:
        Image (B, 3, 64, 64) -> Encoder -> Feature Vector -> Classifier -> Logits
    """
    def __init__(self, feature_dim: int = 512, total_classes: int = 200):
        super().__init__()
        self.encoder = CNNEncoder(feature_dim=feature_dim)
        self.classifier = IncrementalClassifier(
            feature_dim=feature_dim, 
            total_classes=total_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
