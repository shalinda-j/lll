"""
Embedding modules for character and positional encoding.

Layer 4: Character Embedding - map character IDs to learnable vectors
         Add sinusoidal positional encoding
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    Implements the positional encoding from "Attention Is All You Need":
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    This adds position information to the embeddings since transformers
    have no inherent notion of position.
    """

    def __init__(self, embed_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            embed_dim: Dimension of embeddings
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the positional encodings
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim) or (seq_len, embed_dim)

        Returns:
            Tensor with positional encoding added
        """
        if x.dim() == 2:
            # (seq_len, embed_dim) -> add pe directly
            seq_len = x.size(0)
            x = x + self.pe[:seq_len, :]
        else:
            # (batch, seq_len, embed_dim)
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].unsqueeze(0)

        return self.dropout(x)


class CharacterEmbedding(nn.Module):
    """
    Character Embedding Layer (Layer 4).

    Maps each character ID to a learnable float vector.
    Combines learned character embeddings with positional encoding.

    Input: Character IDs (batch, seq_len) or (seq_len,)
    Output: Character vectors (batch, seq_len, embed_dim) or (seq_len, embed_dim)
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embed_dim: int = 256,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize character embedding.

        Args:
            vocab_size: Size of vocabulary (128 for ASCII)
            embed_dim: Dimension of character embeddings
            max_seq_len: Maximum sequence length for positional encoding
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Learned character embeddings
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.char_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed character IDs and add positional encoding.

        Args:
            x: Character IDs tensor of shape (batch, seq_len) or (seq_len,)

        Returns:
            Embedded tensor of shape (batch, seq_len, embed_dim) or (seq_len, embed_dim)
        """
        # Get character embeddings
        embedded = self.char_embedding(x)

        # Add positional encoding
        embedded = self.positional_encoding(embedded)

        return embedded

    def get_embedding_weights(self) -> torch.Tensor:
        """Return the character embedding weights."""
        return self.char_embedding.weight.data


class WordPositionalEncoding(nn.Module):
    """
    Positional encoding for word-level sequences.

    Used in Zone D for sentence/paragraph building.
    Uses the same sinusoidal formula but at word granularity.
    """

    def __init__(self, embed_dim: int = 512, max_words: int = 256, dropout: float = 0.1):
        """
        Initialize word positional encoding.

        Args:
            embed_dim: Dimension of word embeddings
            max_words: Maximum number of words in a sequence
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_words, embed_dim)
        position = torch.arange(0, max_words, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add word positional encoding.

        Args:
            x: Word embeddings of shape (batch, num_words, embed_dim)

        Returns:
            Tensor with positional encoding added
        """
        num_words = x.size(1)
        x = x + self.pe[:num_words, :].unsqueeze(0)
        return self.dropout(x)