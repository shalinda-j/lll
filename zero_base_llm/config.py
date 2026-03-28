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
    vocab_size: int = 128
    ascii_start: int = 0
    ascii_end: int = 127

    # ==================== ZONE B: TRANSFORMER CORE (Layers 4-8) ====================
    embed_dim: int = 256
    max_seq_len: int = 512

    num_heads: int = 8
    head_dim: int = 32
    attention_dropout: float = 0.1   # Reduced from 0.3 — standard transformer value

    ff_dim: int = 1024               # 4× embed_dim (GPT standard)
    ff_dropout: float = 0.1
    ff_activation: str = "gelu"     # GELU: better than ReLU for language models

    num_transformer_blocks: int = 6

    # ==================== ZONE C: WORD BUILDING (Layers 9-12) ====================
    word_boundary_chars: List[int] = field(default_factory=lambda: [32, 44, 46, 33, 63])
    max_word_len: int = 20
    word_embed_dim: int = 512

    context_window: int = 5          # Increased from 3 for wider context

    # ==================== ZONE D: SENTENCE/PARAGRAPH (Layers 13-18) ====================
    temperature: float = 0.8         # Slightly lower than 1.0 for more focused output

    topic_dim: int = 256
    coherence_weight: float = 0.3    # Reduced from 0.5 — let task loss dominate

    max_sentences: int = 10
    completeness_threshold: float = 0.85

    # ==================== ZONE E: OUTPUT (Layers 19-20) ====================
    top_k: int = 50                  # Increased from 10 — more diverse generation
    top_p: float = 0.92              # Slightly higher nucleus mass

    # ==================== ZONE F: SELF-STUDY (Layers 21-22) ====================
    forward_study_weight: float = 0.2   # Reduced from 0.3 — prevent self-study domination
    backward_study_weight: float = 0.2
    consistency_lambda: float = 0.3

    # ==================== TRAINING ====================
    learning_rate: float = 3e-4
    weight_decay: float = 0.1        # Increased from 0.01 — AdamW standard
    grad_clip: float = 1.0
    batch_size: int = 16
    warmup_steps: int = 200          # Linear warmup before cosine decay
    lr_scheduler: str = "cosine_warmup"  # Options: "constant", "cosine", "cosine_warmup"
    label_smoothing: float = 0.1     # Prevents over-confident predictions

    # ==================== MODEL SIZE ====================
    dtype: str = "float32"

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
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        self.head_dim = self.embed_dim // self.num_heads

    @classmethod
    def small(cls) -> "ZeroBaseConfig":
        """Small configuration for testing/debugging (~2M params)."""
        return cls(
            embed_dim=128,
            num_heads=4,
            ff_dim=512,
            ff_activation="gelu",
            num_transformer_blocks=2,
            max_seq_len=256,
            word_embed_dim=256,
            warmup_steps=50,
        )

    @classmethod
    def medium(cls) -> "ZeroBaseConfig":
        """Medium configuration — good quality/speed balance (~10M params)."""
        return cls()

    @classmethod
    def large(cls) -> "ZeroBaseConfig":
        """Large configuration for best quality (~35M params)."""
        return cls(
            embed_dim=512,
            num_heads=8,
            ff_dim=2048,
            ff_activation="gelu",
            num_transformer_blocks=8,
            max_seq_len=1024,
            word_embed_dim=768,
            warmup_steps=500,
            attention_dropout=0.1,
            ff_dropout=0.1,
        )

    @classmethod
    def xl(cls) -> "ZeroBaseConfig":
        """XL configuration — maximum capacity (~110M params, GPT-2 scale)."""
        return cls(
            embed_dim=768,
            num_heads=12,
            ff_dim=3072,
            ff_activation="gelu",
            num_transformer_blocks=12,
            max_seq_len=1024,
            word_embed_dim=1024,
            warmup_steps=1000,
            attention_dropout=0.1,
            ff_dropout=0.1,
            batch_size=8,
            learning_rate=1e-4,
            weight_decay=0.1,
        )
