"""
MAVIS Models Package
====================
Exposes core architecture modules: encoder, classifier, attention, memory, and full MAVIS.
"""

from models.encoder import CNNEncoder
from models.classifier import CosineLinear, IncrementalClassifier
from models.baseline import BaselineCNN
from models.attention import MemoryAttention
from models.memory import MemoryBank
from models.mavis import MAVIS

__all__ = [
    "CNNEncoder",
    "CosineLinear",
    "IncrementalClassifier",
    "BaselineCNN",
    "MemoryAttention",
    "MemoryBank",
    "MAVIS",
]
