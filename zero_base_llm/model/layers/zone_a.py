"""
Zone A: Foundation Layers (0-3)

This module provides a unified interface for the foundation layers,
though the actual implementation is in the tokenizer module.

Layer 0: Binary Foundation - all data as bit arrays
Layer 1: Byte Encoding - binary → byte integers
Layer 2: ASCII Mapping - bytes → ASCII characters
Layer 3: Frequency Sorting - characters sorted by frequency
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from zero_base_llm.tokenizer.tokenizer import BinaryTokenizer


class BinaryFoundation(nn.Module):
    """
    Zone A: Foundation Layers (0-3).

    This module wraps the BinaryTokenizer and provides a PyTorch
    module interface for the binary foundation processing.

    The actual heavy lifting is done by BinaryTokenizer.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        track_frequencies: bool = True
    ):
        """
        Initialize BinaryFoundation.

        Args:
            vocab_size: Size of vocabulary (128 for ASCII)
            track_frequencies: Whether to track character frequencies
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.track_frequencies = track_frequencies

        # The tokenizer handles Layers 0-3
        self.tokenizer = BinaryTokenizer(vocab_size)

    def forward(self, text: str) -> torch.Tensor:
        """
        Process text through foundation layers.

        Args:
            text: Input string

        Returns:
            Tensor of character IDs
        """
        return self.tokenizer.encode(text)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to character IDs."""
        return self.tokenizer.encode(text)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode character IDs to text."""
        return self.tokenizer.decode(ids)

    def get_frequencies(self) -> List[Tuple[str, int]]:
        """Get character frequency counts."""
        return self.tokenizer.get_frequency_sorted_vocab()

    def find_word_boundaries(self, ids: torch.Tensor) -> List[Tuple[int, int]]:
        """Find word boundaries in character IDs."""
        return self.tokenizer.find_word_boundaries(ids)

    @property
    def char_to_id(self):
        """Get character to ID mapping."""
        return self.tokenizer.char_to_id

    @property
    def id_to_char(self):
        """Get ID to character mapping."""
        return self.tokenizer.id_to_char

    def __repr__(self) -> str:
        return f"BinaryFoundation(vocab_size={self.vocab_size})"