"""
MAVIS — Memory Augmented Visual Incremental System (Phase 5 Fusion)
====================================================================
The complete end-to-end architecture that fuses:
  1. CNNEncoder  — extracts features from the current image
  2. MemoryBank  — stores frozen prototypes of past tasks
  3. MemoryAttention — retrieves relevant past features via Multi-Head Attention
  4. Gated Fusion — adaptively blends current and retrieved features
  5. Classifier — maps fused features to class logits

DATA LEAKAGE PREVENTION:
  - Training: model.forward(x, task_id=T) retrieves memories from tasks 0..T-1 ONLY.
  - Evaluation: model.forward(x) (no task_id) retrieves from ALL populated memories.
  - The classifier always outputs logits for ALL 200 classes.
  - Task masking is applied externally in the trainer, NOT inside the model.

OVERFITTING PREVENTION:
  - Gated fusion uses a single linear layer + sigmoid (minimal extra parameters).
  - The gate learns to balance trust between the (possibly degraded) encoder
    and the (frozen, reliable) memory retrieval.
"""

import torch
import torch.nn as nn
from models.encoder import CNNEncoder
from models.memory import MemoryBank
from models.attention import MemoryAttention
from models.classifier import IncrementalClassifier


class MAVIS(nn.Module):
    """
    Full MAVIS architecture.

    Forward pass:
        Image -> Encoder -> current_features
                          -> MemoryAttention(current_features, past_prototypes) -> retrieved_features
                          -> GatedFusion(current, retrieved) -> fused_features
                          -> Classifier -> logits [200 classes]
    """
    def __init__(self, config):
        super().__init__()
        feature_dim = config.encoder.feature_dim
        num_tasks = config.dataset.num_tasks
        classes_per_task = config.dataset.classes_per_task
        total_classes = config.dataset.num_classes
        num_heads = config.attention.num_heads
        dropout_rate = config.encoder.dropout_rate

        # 1. Encoder (same Wide ResNet-18 used in Phases 2 & 3)
        self.encoder = CNNEncoder(feature_dim=feature_dim, dropout_rate=dropout_rate)

        # 2. Memory Bank (frozen prototype storage — NOT trained)
        self.memory_bank = MemoryBank(feature_dim, num_tasks, classes_per_task)

        # 3. Attention retrieval
        self.attention = MemoryAttention(feature_dim, num_heads=num_heads, dropout=0.1)

        # 4. Gated Fusion
        #    g = sigmoid(W_g * [current ; retrieved] + b_g)
        #    fused = g * current + (1 - g) * retrieved
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        # 5. Classifier (same single-layer head used in Phases 2 & 3)
        self.classifier = IncrementalClassifier(feature_dim=feature_dim, total_classes=total_classes)

    def forward(self, x, task_id=None):
        """
        Args:
            x: input images [B, 3, 64, 64]
            task_id: (int or None)
                - During TRAINING: pass the current task_id.
                  Memory retrieval uses tasks 0..task_id-1 only.
                - During EVALUATION: pass None.
                  Memory retrieval uses all populated tasks.

        Returns:
            logits: [B, 200] — raw logits for all 200 classes.
                    Task masking (if needed) is done externally in the trainer.
        """
        # 1. Extract current features
        current_features = self.encoder(x)

        # 2. Retrieve past features from memory
        if task_id is not None:
            # TRAINING: only retrieve from completed past tasks (DATA LEAKAGE PREVENTION)
            past_memories = self.memory_bank.get_past_memories(task_id)
        else:
            # EVALUATION: retrieve from everything stored so far
            past_memories = self.memory_bank.get_all_populated_memories()

        retrieved_features = self.attention(current_features, past_memories)

        # 3. Gated Fusion
        concat = torch.cat([current_features, retrieved_features], dim=1)  # [B, 2*D]
        gate = self.fusion_gate(concat)  # [B, D], values in (0, 1)
        fused_features = gate * current_features + (1.0 - gate) * retrieved_features

        # 4. Classify from fused features
        logits = self.classifier(fused_features)

        return logits
