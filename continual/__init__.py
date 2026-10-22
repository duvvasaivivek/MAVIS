"""
MAVIS Continual Learning Package
================================
Exposes continual learning trainers and parameter consolidation modules.
"""

from continual.ewc import OnlineEWC, compute_fisher, snapshot_params
from continual.trainer import ContinualTrainer
from continual.mavis_trainer import MAVISTrainer

__all__ = [
    "OnlineEWC",
    "compute_fisher",
    "snapshot_params",
    "ContinualTrainer",
    "MAVISTrainer",
]
