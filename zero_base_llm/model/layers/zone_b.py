"""
Zone B: Transformer Core (Layers 4-8)

This module implements the core transformer architecture:

Layer 4: Character Embedding - map IDs to vectors + positional encoding
Layer 5: Self-Attention Block 1 - single head attention
Layer 6: Multi-Head Attention Block 2 - 8 parallel heads
Layer 7: Feed Forward Network - two-layer MLP
Layer 8: Residual Connection + Layer Normalization

The TransformerBlock combines Layers 5-8 into a single reusable unit.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from ..modules.embeddings import CharacterEmbedding, PositionalEncoding
from ..modules.attention import SelfAttention, MultiHeadAttention, create_causal_mask
from ..modules.normalization import LayerNorm, ResidualConnection, FeedForward


class TransformerBlock(nn.Module):
    """
    Single Transformer Block combining Layers 5-8.

    Structure (Pre-LN variant):
    1. LayerNorm → Multi-Head Attention → Residual Add
    2. LayerNorm → Feed-Forward Network → Residual Add

    This block is repeated N times to build the transformer core.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize TransformerBlock.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout rate
            activation: Activation function for FFN
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Layer 5-6: Multi-Head Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Layer 7: Feed-Forward Network
        self.ffn = FeedForward(embed_dim, ff_dim, dropout, activation)

        # Layer 8: Layer Normalization (for pre-norm)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

        # Dropout for residual connections
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Pre-norm attention block
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, mask)
        x = x + self.dropout(attn_out)

        # Pre-norm FFN block
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, attn_weights


class TransformerCore(nn.Module):
    """
    Zone B: Transformer Core (Layers 4-8).

    Combines character embedding with multiple transformer blocks.

    Input: Character IDs (batch, seq_len)
    Output: Hidden states (batch, seq_len, embed_dim)
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embed_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_blocks: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize TransformerCore.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            num_blocks: Number of transformer blocks
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len

        # Layer 4: Character Embedding
        self.embedding = CharacterEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # Layers 5-8: Transformer Blocks (repeated)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, activation)
            for _ in range(num_blocks)
        ])

        # Final layer norm (for pre-norm architecture)
        self.final_norm = LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through transformer core.

        Args:
            input_ids: Character IDs (batch, seq_len)
            mask: Optional attention mask (if None, creates causal mask)
            use_causal_mask: Whether to use causal (autoregressive) masking

        Returns:
            Tuple of (hidden_states, list_of_attention_weights)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create causal mask if needed
        if mask is None and use_causal_mask:
            mask = create_causal_mask(seq_len, device)

        # Layer 4: Embed characters
        x = self.embedding(input_ids)

        # Process through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            attention_weights.append(attn_weights)

        # Final layer norm
        x = self.final_norm(x)

        return x, attention_weights

    def get_embeddings(self) -> torch.Tensor:
        """Get the character embedding weights."""
        return self.embedding.get_embedding_weights()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerEncoder(TransformerCore):
    """
    Transformer Encoder variant (bidirectional attention).

    Unlike the autoregressive TransformerCore, this allows
    attention to all positions (no causal mask).
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with bidirectional attention.

        Args:
            input_ids: Character IDs
            mask: Optional attention mask

        Returns:
            Tuple of (hidden_states, attention_weights)
        """
        # Call parent with use_causal_mask=False
        return super().forward(input_ids, mask, use_causal_mask=False)