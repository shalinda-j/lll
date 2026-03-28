"""
Attention mechanisms for the Transformer Core.

Layer 5: Self-Attention Block 1 - Single head, d_k = 64
Layer 6: Multi-Head Attention Block 2 - 8 parallel heads, d_k = 32 each

Implements scaled dot-product attention:
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    This is the core attention operation used in transformers.
    """

    def __init__(self, dropout: float = 0.1):
        """
        Initialize attention.

        Args:
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor (batch, heads, seq_len, d_k)
            key: Key tensor (batch, heads, seq_len, d_k)
            value: Value tensor (batch, heads, seq_len, d_v)
            mask: Optional mask tensor (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

        Returns:
            Tuple of (output, attention_weights)
        """
        d_k = query.size(-1)

        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class SelfAttention(nn.Module):
    """
    Single-Head Self-Attention (Layer 5).

    Implements self-attention where Q, K, V are all derived from the same input.
    Single head with d_k = 64 (default).

    Input: (batch, seq_len, embed_dim)
    Output: (batch, seq_len, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        head_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize single-head self-attention.

        Args:
            embed_dim: Input embedding dimension
            head_dim: Dimension of attention head
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)

        # Output projection (head_dim -> embed_dim)
        self.W_O = nn.Linear(head_dim, embed_dim, bias=False)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)

        # Dropout for output
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply single-head self-attention.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_Q(x)  # (batch, seq_len, head_dim)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for attention: (batch, 1, seq_len, head_dim)
        # Adding head dimension for compatibility with multi-head attention code
        Q = Q.unsqueeze(1)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        # Compute attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)

        # Remove head dimension
        attn_output = attn_output.squeeze(1)

        # Project back to embed_dim
        output = self.W_O(attn_output)
        output = self.dropout(output)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Layer 6).

    Implements multi-head attention with parallel attention heads.
    Each head learns different aspects of the input.

    Default: 8 heads with d_k = 32 each (total = 256 = embed_dim)

    Input: (batch, seq_len, embed_dim)
    Output: (batch, seq_len, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Input embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V (all heads at once)
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)

        # Dropout for output
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_Q(x)  # (batch, seq_len, embed_dim)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)

        # Concatenate heads: (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        # Output projection
        output = self.W_O(attn_output)
        output = self.dropout(output)

        return output, attention_weights


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism.

    Used when query comes from one sequence and key/value from another.
    Useful for encoder-decoder attention and certain self-study mechanisms.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention.

        Args:
            query: Query tensor (batch, query_len, embed_dim)
            key_value: Key and value tensor (batch, kv_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)

        # Project
        Q = self.W_Q(query)
        K = self.W_K(key_value)
        V = self.W_V(key_value)

        # Reshape for multi-head
        query_len = query.size(1)
        kv_len = key_value.size(1)

        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.embed_dim
        )

        # Output projection
        output = self.W_O(attn_output)
        output = self.dropout(output)

        return output, attention_weights


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) attention mask.

    This mask prevents attending to future positions, which is essential
    for autoregressive language modeling.

    Args:
        seq_len: Sequence length
        device: Torch device

    Returns:
        Mask tensor of shape (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return mask