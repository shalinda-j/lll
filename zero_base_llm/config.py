"""
Central configuration for the Zero-Base LLM architecture.

This module defines all hyperparameters for the 22-layer architecture
across 6 zones: Foundation, Transformer Core, Word Building,
Sentence/Paragraph Building, Output, and Self-Study.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ZeroBaseConfig:
    """Configuration for all 22 layers of the Zero-Base LLM."""

    # ==================== ZONE A: FOUNDATION (Layers 0-3) ====================
    # Layer 0: Binary Foundation
    # Layer 1: Byte Encoding
    # Layer 2: ASCII Mapping
    vocab_size: int = 128  # ASCII printable range (0-127)
    ascii_start: int = 0   # Start of ASCII range
    ascii_end: int = 127   # End of ASCII range

    # ==================== ZONE B: TRANSFORMER CORE (Layers 4-8) ====================
    # Layer 4: Character Embedding
    embed_dim: int = 256           # Character embedding dimension
    max_seq_len: int = 512         # Maximum sequence length

    # Layer 5-6: Attention
    num_heads: int = 8             # Number of attention heads
    head_dim: int = 32             # Dimension per head (256/8)
    attention_dropout: float = 0.3 # Dropout for attention weights (increased to prevent overfitting)

    # Layer 7: Feed Forward Network
    ff_dim: int = 512              # Hidden dimension in FFN
    ff_dropout: float = 0.3        # Dropout for FFN (increased to prevent overfitting)

    # Transformer block configuration
    num_transformer_blocks: int = 4  # Number of transformer blocks to stack

    # ==================== ZONE C: WORD BUILDING (Layers 9-12) ====================
    # Layer 9: Word boundary detection
    word_boundary_chars: List[int] = field(default_factory=lambda: [32, 44, 46, 33, 63])  # space, comma, period, !, ?
    max_word_len: int = 20         # Maximum characters per word
    word_embed_dim: int = 512      # Word-level embedding dimension

    # Layer 12: Context window
    context_window: int = 3        # Words on each side for context

    # ==================== ZONE D: SENTENCE/PARAGRAPH (Layers 13-18) ====================
    # Layer 16: Sentence completion
    temperature: float = 1.0       # Sampling temperature

    # Layer 17: Multi-sentence coherence
    topic_dim: int = 256           # Topic vector dimension
    coherence_weight: float = 0.5  # Weight for coherence loss

    # Layer 18: Paragraph assembly
    max_sentences: int = 10        # Maximum sentences per paragraph
    completeness_threshold: float = 0.85  # Stop generation when score > threshold

    # ==================== ZONE E: OUTPUT (Layers 19-20) ====================
    # Layer 20: Sampling strategies
    top_k: int = 10                # Top-k sampling
    top_p: float = 0.9             # Nucleus sampling probability mass

    # ==================== ZONE F: SELF-STUDY (Layers 21-22) ====================
    # Layer 21: Forward self-study
    forward_study_weight: float = 0.3  # Weight for forward study loss

    # Layer 22: Backward self-study
    backward_study_weight: float = 0.3  # Weight for backward study loss
    consistency_lambda: float = 0.5     # Lambda for consistency loss

    # ==================== TRAINING ====================
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 16

    # ==================== MODEL SIZE ====================
    dtype: str = "float32"         # Options: "float32", "float16", "bfloat16"

    def get_dtype(self):
        """Return the PyTorch dtype."""
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)

    def __post_init__(self):
        """Validate configuration."""
        # Ensure embed_dim is divisible by num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        # Set head_dim based on embed_dim and num_heads
        self.head_dim = self.embed_dim // self.num_heads

    @classmethod
    def small(cls) -> "ZeroBaseConfig":
        """Small configuration for testing/debugging."""
        return cls(
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            num_transformer_blocks=2,
            max_seq_len=256,
            word_embed_dim=256,
        )

    @classmethod
    def medium(cls) -> "ZeroBaseConfig":
        """Medium configuration for good quality output."""
        return cls()

    @classmethod
    def large(cls) -> "ZeroBaseConfig":
        """Large configuration for best quality."""
        return cls(
            embed_dim=512,
            num_heads=8,
            ff_dim=1024,
            num_transformer_blocks=8,
            max_seq_len=1024,
            word_embed_dim=768,
        )