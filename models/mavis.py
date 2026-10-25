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
import torch.nn.functional as F
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
        self.feature_dim = feature_dim
        num_tasks = config.dataset.num_tasks
        classes_per_task = config.dataset.classes_per_task
        total_classes = config.dataset.num_classes
        num_heads = config.attention.num_heads
        dropout_rate = config.encoder.dropout_rate

        # 1. Encoder (same Wide ResNet-18 used in Phases 2 & 3)
        base_width = getattr(config.encoder, "base_width", 128)
        self.encoder = CNNEncoder(feature_dim=feature_dim, dropout_rate=dropout_rate, base_width=base_width)

        # Ablation switch: when False, the attention retrieval and gated fusion are bypassed and the
        # classifier sees the raw encoder feature. Everything else (memory bank, replay, NCM,
        # re-alignment, distillation, EWC) is unchanged. Used to measure what the forward-pass
        # memory path itself contributes.
        self.use_attention = bool(getattr(config.attention, "enabled", True))
        # Per-forward diagnostics (mean L2 norm of live feature, retrieved feature, mean gate value).
        self.last_norms = None

        # 2. Memory Bank (frozen multi-prototype storage — NOT trained)
        prototypes_per_class = getattr(config.memory, "prototypes_per_class", 3)
        self.memory_bank = MemoryBank(feature_dim, num_tasks, classes_per_task, prototypes_per_class=prototypes_per_class)

        # 3. Attention retrieval (with temperature sharpening)
        attn_temp = getattr(config.attention, "temperature", 0.5)
        self.attention = MemoryAttention(feature_dim, num_heads=num_heads, dropout=0.1, temperature=attn_temp)

        # 4. Gated Fusion
        #    g = sigmoid(W_g * [current ; retrieved] + b_g)
        #    fused = g * current + (1 - g) * retrieved
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        # Initialize gate weights and bias to 0 so initial gate is exactly 0.5 (balanced 50/50)
        nn.init.zeros_(self.fusion_gate[0].weight)
        nn.init.zeros_(self.fusion_gate[0].bias)

        # 5. Classifier (balanced CosineLinear head with Cosine Margin)
        margin = getattr(config.classifier, "margin", 0.20)
        self.classifier = IncrementalClassifier(feature_dim=feature_dim, total_classes=total_classes, margin=margin)

    def forward(self, x, task_id=None, targets=None, return_features=False, return_spatials=False):
        """
        Args:
            x: input images [B, 3, 64, 64]
            task_id: (int or None)
                - During TRAINING: pass the current task_id.
                  Memory retrieval uses tasks 0..task_id-1 plus running prototypes for task_id.
                - During EVALUATION: pass None.
                  Memory retrieval uses all locked memories.
            targets: (torch.Tensor or None) Ground truth class targets for additive margin
            return_features: bool, whether to return encoder and fused features
            return_spatials: bool, whether to return intermediate convolutional feature maps

        Returns:
            logits: [B, 200]
            (optional) current_features, fused_features, spatials
        """
        # 1. Extract current features
        if return_spatials:
            current_features, spatials = self.encoder(x, return_spatial=True)
        else:
            current_features = self.encoder(x)
            spatials = None

        # 2. Retrieve past/active features from memory
        if task_id is not None:
            # TRAINING: retrieve from past completed tasks + current task running prototypes
            memories = self.memory_bank.get_training_memories(task_id)
        else:
            # EVALUATION: retrieve from all locked populated memories
            memories = self.memory_bank.get_all_populated_memories()

        if self.use_attention:
            retrieved_features = self.attention(current_features, memories)

            # 3. Gated Fusion
            # Clamped gate (0.05 .. 0.95) allows high fidelity feature routing while preserving memory retrieval
            concat = torch.cat([current_features, retrieved_features], dim=1)  # [B, 2*D]
            raw_gate = self.fusion_gate(concat)  # [B, D], values in (0, 1)
            gate = 0.05 + 0.90 * raw_gate
            fused_features = gate * current_features + (1.0 - gate) * retrieved_features
            with torch.no_grad():
                self.last_norms = {
                    "f_norm": current_features.float().norm(dim=1).mean().item(),
                    "r_norm": retrieved_features.float().norm(dim=1).mean().item(),
                    "gate_mean": gate.float().mean().item(),
                    # share of the fused vector's squared norm that comes from the retrieved branch
                    "r_share": ((1.0 - gate) * retrieved_features).float().norm(dim=1).pow(2).mean().item()
                               / max(fused_features.float().norm(dim=1).pow(2).mean().item(), 1e-12),
                }
        else:
            # Ablation: no memory path in the forward pass.
            fused_features = current_features
            with torch.no_grad():
                self.last_norms = {
                    "f_norm": current_features.float().norm(dim=1).mean().item(),
                    "r_norm": 0.0, "gate_mean": 1.0, "r_share": 0.0,
                }

        # 4. Classify from fused features (targets passed for cosine margin when training)
        logits = self.classifier(fused_features, targets=targets)

        if return_spatials:
            return logits, current_features, fused_features, spatials
        if return_features:
            return logits, current_features, fused_features
        return logits

    def fuse_features(self, current_features: torch.Tensor, memories: torch.Tensor = None, chunk_size: int = 64) -> torch.Tensor:
        """
        Applies Memory Attention and Gated Fusion directly to feature vectors.
        Harmonizes prototype representations with the inference feature distribution.
        Uses query chunking to guarantee bounded VRAM (<300MB) even with thousands of queries.
        """
        if not self.use_attention:
            # Ablation: replayed prototypes reach the classifier exactly as live features do (no fusion).
            return F.normalize(current_features, p=2, dim=-1)

        if memories is None:
            memories = self.memory_bank.get_all_populated_memories()

        # Fixed 2026-10-25. Callers hand this method unit-norm prototypes, but the attention and the
        # fusion gate were trained on live encoder features, which leave a LayerNorm at length
        # ~sqrt(D). Feeding the gate a query ~22x shorter than it expects changes the gate it
        # produces, and once the retrieval path stopped being suppressed (see the reconstruction
        # loss in MAVISTrainer) the retrieved branch would have swamped the query outright. Put the
        # query on the scale the gate was trained for.
        current_features = F.normalize(current_features, p=2, dim=-1) * (self.feature_dim ** 0.5)

        batch_size = current_features.size(0)
        if batch_size > chunk_size:
            fused_list = []
            for i in range(0, batch_size, chunk_size):
                c_chunk = current_features[i:i + chunk_size]
                ret_chunk = self.attention(c_chunk, memories, chunk_size=chunk_size)
                concat = torch.cat([c_chunk, ret_chunk], dim=1)
                raw_gate = self.fusion_gate(concat)
                gate = 0.05 + 0.90 * raw_gate
                fused = gate * c_chunk + (1.0 - gate) * ret_chunk
                fused_list.append(F.normalize(fused, p=2, dim=-1))
            return torch.cat(fused_list, dim=0)

        retrieved_features = self.attention(current_features, memories, chunk_size=chunk_size)
        concat = torch.cat([current_features, retrieved_features], dim=1)
        raw_gate = self.fusion_gate(concat)
        gate = 0.05 + 0.90 * raw_gate
        fused = gate * current_features + (1.0 - gate) * retrieved_features
        return F.normalize(fused, p=2, dim=-1)
