"""
Main Model: Zero-Base LLM (22 Layers)

This module integrates all 6 zones into a complete language model:

Zone A: Foundation Layers (0-3) - Binary → ASCII
Zone B: Transformer Core (Layers 4-8) - Attention & FFN
Zone C: Word Building (Layers 9-12) - Character to Word
Zone D: Sentence/Paragraph (Layers 13-18) - Sentence assembly
Zone E: Output (Layers 19-20) - Projection & Sampling
Zone F: Self-Study (Layers 21-22) - Self-improvement
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any

from ..config import ZeroBaseConfig
from .layers.zone_a import BinaryFoundation
from .layers.zone_b import TransformerCore
from .layers.zone_c import WordBuilder
from .layers.zone_d import SentenceBuilder
from .layers.zone_e import OutputLayer
from .layers.zone_f import SelfStudySystem


class ZeroBaseLLM(nn.Module):
    """
    Zero-Base LLM: A complete 22-layer language model built from binary foundation.

    This model processes text through all 6 zones:
    1. Zone A (Layers 0-3): Binary Foundation → ASCII encoding
    2. Zone B (Layers 4-8): Transformer Core with attention
    3. Zone C (Layers 9-12): Character clustering → Word building
    4. Zone D (Layers 13-18): Sentence & Paragraph assembly
    5. Zone E (Layers 19-20): Output projection & sampling
    6. Zone F (Layers 21-22): Self-study for improvement

    No external training data required - the model self-improves through
    bidirectional self-study (forward prediction + backward consistency).
    """

    def __init__(self, config: Optional[ZeroBaseConfig] = None):
        """
        Initialize ZeroBaseLLM.

        Args:
            config: Model configuration (uses defaults if None)
        """
        super().__init__()
        self.config = config or ZeroBaseConfig()

        # ==================== ZONE A: Foundation (Layers 0-3) ====================
        self.foundation = BinaryFoundation(
            vocab_size=self.config.vocab_size
        )

        # ==================== ZONE B: Transformer Core (Layers 4-8) ====================
        self.transformer_core = TransformerCore(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            num_blocks=self.config.num_transformer_blocks,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.attention_dropout,
            activation=getattr(self.config, "ff_activation", "gelu"),
        )

        # ==================== ZONE C: Word Building (Layers 9-12) ====================
        self.word_builder = WordBuilder(
            char_embed_dim=self.config.embed_dim,
            word_embed_dim=self.config.word_embed_dim,
            context_window=self.config.context_window,
            dropout=self.config.ff_dropout
        )

        # ==================== ZONE D: Sentence/Paragraph (Layers 13-18) ====================
        self.sentence_builder = SentenceBuilder(
            word_embed_dim=self.config.word_embed_dim,
            vocab_size=self.config.vocab_size,
            max_sentences=self.config.max_sentences,
            temperature=self.config.temperature,
            dropout=self.config.ff_dropout
        )

        # ==================== ZONE E: Output (Layers 19-20) ====================
        # Word-level output layer (for paragraph-level generation)
        self.output_layer = OutputLayer(
            hidden_dim=self.config.word_embed_dim,
            vocab_size=self.config.vocab_size,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p
        )

        # Character-level output layer (for character-level training)
        self.char_output_layer = OutputLayer(
            hidden_dim=self.config.embed_dim,
            vocab_size=self.config.vocab_size,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p
        )

        # ==================== ZONE F: Self-Study (Layers 21-22) ====================
        # Use embed_dim for character-level self-study
        self.self_study = SelfStudySystem(
            embed_dim=self.config.embed_dim,
            vocab_size=self.config.vocab_size,
            hidden_dim=self.config.embed_dim // 2,
            forward_weight=self.config.forward_study_weight,
            backward_weight=self.config.backward_study_weight,
            consistency_lambda=self.config.consistency_lambda,
            dropout=self.config.ff_dropout
        )

        # Apply weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        GPT-style weight initialization for stable deep network training.

        - Linear layers: N(0, 0.02)
        - Residual projection layers scaled by 1/sqrt(2 * num_layers)
          to prevent variance explosion through residual stacks.
        - Embeddings: N(0, 0.02)
        - LayerNorm: weight=1, bias=0
        """
        num_layers = self.config.num_transformer_blocks
        residual_scale = (2 * num_layers) ** -0.5

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # Scale down projection layers inside residual paths
                if any(tag in name for tag in ["proj", "out_proj", "c_proj"]):
                    module.weight.data.mul_(residual_scale)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_self_study: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all 22 layers.

        Args:
            input_ids: Character IDs (batch, seq_len)
            targets: Optional target tokens for loss computation
            use_self_study: Whether to apply self-study layers

        Returns:
            Dictionary with logits, losses, and auxiliary outputs
        """
        batch_size, seq_len = input_ids.shape

        # ==================== ZONE B: Transformer Core ====================
        # Layers 4-8: Character-level processing
        char_hidden, attention_weights = self.transformer_core(input_ids)

        # ==================== ZONE C: Word Building ====================
        # Layers 9-12: Build word representations
        word_vectors_list, word_boundaries = self.word_builder(
            char_hidden, input_ids
        )

        # Process word vectors for sentence building
        # (Handle variable number of words per batch)
        outputs = {
            "char_hidden": char_hidden,
            "attention_weights": attention_weights,
            "word_boundaries": word_boundaries,
        }

        # For simplicity in the forward pass, process each batch item
        all_logits = []
        all_coherence = []

        for word_vectors in word_vectors_list:
            if word_vectors.size(0) == 0:
                # No words detected, use character hidden state
                word_vectors = char_hidden[:, -1:, :].mean(dim=1, keepdim=True)

            # ==================== ZONE D: Sentence/Paragraph ====================
            # Layers 13-18: Sentence building
            logits, coherence = self.sentence_builder(word_vectors)
            all_logits.append(logits)
            all_coherence.append(coherence)

        # Stack outputs (pad if necessary for batching)
        # For simplicity, use the first batch item's logits
        # The sentence_builder already produces vocab logits, so we use those directly
        if all_logits:
            # sentence_builder returns logits over vocabulary
            final_logits = all_logits[0]
        else:
            # Fallback: use char_hidden directly through output projection
            final_logits = self.output_layer(char_hidden)

        outputs["logits"] = final_logits
        outputs["coherence_scores"] = torch.cat(all_coherence, dim=0) if all_coherence else None

        # Compute loss if targets provided
        # For character-level training, use the transformer output directly
        if targets is not None:
            # Use character-level logits for training (maintains sequence length)
            char_logits = self.char_output_layer(char_hidden)
            loss = self._compute_loss(char_logits, targets)
            outputs["loss"] = loss

        # ==================== ZONE F: Self-Study ====================
        # Layers 21-22: Self-improvement
        if use_self_study and self.training:
            # Get hidden state for self-study
            hidden_for_study = char_hidden[:, -1, :]  # Last position

            continuation_logits, consistency_score, backward_loss = self.self_study(
                hidden_for_study.unsqueeze(1),
                char_hidden[:, 0:1, :]  # Use start as pseudo-input
            )

            outputs["continuation_logits"] = continuation_logits
            outputs["consistency_score"] = consistency_score
            outputs["self_study_loss"] = backward_loss

        return outputs

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model logits
            targets: Target tokens

        Returns:
            Loss tensor
        """
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=-100
        )

        return loss

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to character IDs.

        Args:
            text: Input string

        Returns:
            Character ID tensor
        """
        return self.foundation.encode(text)

    def decode(self, ids: torch.Tensor) -> str:
        """
        Decode character IDs to text.

        Args:
            ids: Character ID tensor

        Returns:
            Decoded string
        """
        return self.foundation.decode(ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.9,
        strategy: str = "nucleus"
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            strategy: Sampling strategy

        Returns:
            Generated text
        """
        self.eval()

        # Encode prompt
        input_ids = self.encode(prompt)
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension

        generated = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(input_ids, use_self_study=False)
                logits = outputs["logits"]

                # Get logits for next token (last position)
                next_logits = logits[:, -1, :]

                # Sample next token
                next_token = self.output_layer.sampler.sample(
                    next_logits,
                    strategy=strategy,
                    temperature=temperature,
                    p=top_p,
                    k=top_k
                )

                generated.append(next_token.item())

                # Append to input (next_token is shape [batch], need [batch, 1])
                next_token_2d = next_token.unsqueeze(-1)  # (batch, 1)
                input_ids = torch.cat([input_ids, next_token_2d], dim=1)

                # Truncate if too long
                if input_ids.size(1) > self.config.max_seq_len:
                    input_ids = input_ids[:, -self.config.max_seq_len:]

        # Decode generated tokens
        generated_text = self.decode(torch.tensor(generated))

        return prompt + generated_text

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """
        Get estimated model size in megabytes.

        Returns:
            Size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)

    def save(self, path: str):
        """
        Save model weights.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "config": self.config.__dict__,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str) -> "ZeroBaseLLM":
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = ZeroBaseConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def __repr__(self) -> str:
        return (
            f"ZeroBaseLLM(\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  embed_dim={self.config.embed_dim},\n"
            f"  num_heads={self.config.num_heads},\n"
            f"  num_blocks={self.config.num_transformer_blocks},\n"
            f"  parameters={self.count_parameters():,},\n"
            f"  size={self.get_model_size_mb():.2f}MB\n"
            f")"
        )