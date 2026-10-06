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
import torch.nn.functional as F
import math

class ManualMultiHeadAttention(nn.Module):
    """
    Pure PyTorch implementation of Multi-Head Attention to bypass Triton/FlashAttention C++ compilers.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # CRITICAL HPC FIX: Force FP32 to prevent PyTorch from trying to compile Triton FP16 TensorCore kernels
        with torch.amp.autocast('cuda', enabled=False):
            orig_dtype = query.dtype
            query = query.float()
            key = key.float()
            value = value.float()

            B, T_q, C = query.size()
            _, T_k, _ = key.size()

            # Linear projections
            q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_q, hd]
            k = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, T_k, hd]
            v = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_k, hd]

            # CRITICAL HPC FIX: 
            # 4D torch.matmul triggers the Triton C++ compiler on PyTorch 2.14.
            # By reshaping to 3D and using torch.bmm, we force PyTorch to use pre-compiled cuBLAS.
            q_3d = q.reshape(B * self.num_heads, T_q, self.head_dim)
            k_3d = k.reshape(B * self.num_heads, T_k, self.head_dim).transpose(1, 2)
            v_3d = v.reshape(B * self.num_heads, T_k, self.head_dim)

            # Scaled dot-product attention (3D bmm)
            scores_3d = torch.bmm(q_3d, k_3d) / math.sqrt(self.head_dim)  # [B*H, T_q, T_k]
            attn_3d = F.softmax(scores_3d, dim=-1)
            attn_3d = self.dropout(attn_3d)

            # Apply attention to values (3D bmm)
            out_3d = torch.bmm(attn_3d, v_3d)  # [B*H, T_q, hd]
            
            # Reshape back to 4D
            out = out_3d.reshape(B, self.num_heads, T_q, self.head_dim)

            # Final linear projection
            out = out.transpose(1, 2).contiguous().view(B, T_q, C)  # [B, T_q, C]

            # Restore original dtype
            out = out.to(orig_dtype)

            return self.out_proj(out), attn_3d.reshape(B, self.num_heads, T_q, T_k)


class MemoryAttention(nn.Module):
    """
    Multi-Head Attention that retrieves relevant features from frozen memory prototypes.
    """
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.mha = ManualMultiHeadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
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
        # Use .repeat() instead of .expand() to prevent C++ SDPA backend segfaults on zero-stride tensors
        kv = memory_prototypes.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, N, D]

        # Force pure PyTorch math (Custom MHA)
        attn_output, _ = self.mha(query, kv, kv)  # [B, 1, D]
        attn_output = attn_output.squeeze(1)       # [B, D]

        return self.layer_norm(attn_output)
