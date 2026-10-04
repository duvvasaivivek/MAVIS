"""
Multi-Head Attention for Memory Retrieval (Phase 4)
=====================================================
Retrieves relevant past-task features from the Memory Bank.

Query:  Current image features from the encoder
Key/Value: Frozen prototypes stored in the Memory Bank

DATA LEAKAGE PREVENTION:
  - Attention is applied to PAST memories only (never the current task's prototypes).
  - When no past memories exist (Task 1), returns a learned zero-like output
    (not raw zeros, which would create a distribution shift between tasks).

OVERFITTING PREVENTION:
  - Dropout on attention weights to prevent over-reliance on specific prototypes.
  - LayerNorm on output for stable gradients.
"""

import torch
import torch.nn as nn


class MemoryAttention(nn.Module):
    """
    Multi-Head Attention that retrieves relevant features from frozen memory prototypes.
    """
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(feature_dim)

        # Learned "no-memory" embedding for Task 1 (when memory bank is empty).
        # This avoids the distribution shift of returning raw zeros.
        self.no_memory_embed = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.normal_(self.no_memory_embed, mean=0, std=0.01)

    def forward(self, current_features, memory_prototypes):
        """
        Args:
            current_features: [B, feature_dim] — live encoder output
            memory_prototypes: [N, feature_dim] — frozen past prototypes, or None

        Returns:
            [B, feature_dim] — attention-retrieved features
        """
        batch_size = current_features.size(0)

        if memory_prototypes is None:
            # Task 1: no past memories. Return the learned no-memory embedding.
            return self.no_memory_embed.expand(batch_size, -1)

        # Reshape for MHA: query=[B,1,D], key/value=[B,N,D]
        query = current_features.unsqueeze(1)                      # [B, 1, D]
        kv = memory_prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, D]

        attn_output, _ = self.mha(query, kv, kv)  # [B, 1, D]
        attn_output = attn_output.squeeze(1)       # [B, D]

        return self.layer_norm(attn_output)
