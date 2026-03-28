"""
Text Generation Module for Zero-Base LLM.

This module provides utilities for autoregressive text generation
with various sampling strategies.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Callable, Generator
from dataclasses import dataclass

from ..model.model import ZeroBaseLLM
from ..config import ZeroBaseConfig


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 10
    top_p: float = 0.9
    strategy: str = "nucleus"  # "greedy", "top_k", "nucleus", "temperature"
    stop_tokens: Optional[List[int]] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0


class TextGenerator:
    """
    Text generator for Zero-Base LLM.

    Supports multiple generation strategies:
    - Greedy decoding
    - Top-k sampling
    - Nucleus (top-p) sampling
    - Temperature-based sampling
    - Beam search (planned)
    """

    def __init__(
        self,
        model: ZeroBaseLLM,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize text generator.

        Args:
            model: ZeroBaseLLM model
            config: Generation configuration
        """
        self.model = model
        self.config = config or GenerationConfig()

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        return self.model.encode(text)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.model.decode(ids)

    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits.

        Args:
            logits: Logits (vocab_size,)
            generated_ids: Previously generated token IDs
            penalty: Penalty factor

        Returns:
            Penalized logits
        """
        if penalty == 1.0 or len(generated_ids) == 0:
            return logits

        # Get unique tokens
        unique_ids = torch.unique(generated_ids)

        # Penalize
        logits[unique_ids] = logits[unique_ids] / penalty

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            config: Override generation config

        Returns:
            Generated text (including prompt)
        """
        cfg = config or self.config
        self.model.eval()

        # Encode prompt
        input_ids = self.encode(prompt)
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension

        generated_ids = input_ids.clone()
        new_tokens = []

        for _ in range(cfg.max_new_tokens):
            # Forward pass
            outputs = self.model(generated_ids, use_self_study=False)
            logits = outputs["logits"]

            # Get next token logits
            next_logits = logits[0, -1, :].clone()

            # Apply repetition penalty
            if len(new_tokens) > 0:
                new_tokens_tensor = torch.tensor(new_tokens, device=next_logits.device)
                next_logits = self.apply_repetition_penalty(
                    next_logits, new_tokens_tensor, cfg.repetition_penalty
                )

            # Apply temperature
            if cfg.temperature != 1.0:
                next_logits = next_logits / cfg.temperature

            # Sample next token
            if cfg.strategy == "greedy":
                next_token = torch.argmax(next_logits).unsqueeze(0)
            elif cfg.strategy == "top_k":
                top_k_logits, top_k_indices = torch.topk(next_logits, cfg.top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_k_indices[idx]
            elif cfg.strategy == "nucleus":
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > cfg.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                sorted_logits[sorted_indices_to_remove] = float('-inf')
                probs = F.softmax(sorted_logits, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = sorted_indices[idx]
            else:  # temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            next_token = next_token.squeeze()
            new_tokens.append(next_token.item())

            # Check stop tokens
            if cfg.stop_tokens and next_token.item() in cfg.stop_tokens:
                break

            # Append to generated
            generated_ids = torch.cat([
                generated_ids,
                next_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)

            # Truncate if needed
            if generated_ids.size(1) > self.model.config.max_seq_len:
                generated_ids = generated_ids[:, -self.model.config.max_seq_len:]

        # Decode generated text
        generated_text = self.decode(generated_ids[0])
        return generated_text

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Generator[str, None, None]:
        """
        Generate text as a stream, yielding each new token.

        Args:
            prompt: Input prompt
            config: Generation config

        Yields:
            Generated text after each token
        """
        cfg = config or self.config
        self.model.eval()

        input_ids = self.encode(prompt).unsqueeze(0)
        generated_ids = input_ids.clone()

        for _ in range(cfg.max_new_tokens):
            outputs = self.model(generated_ids, use_self_study=False)
            logits = outputs["logits"][0, -1, :]

            if cfg.temperature != 1.0:
                logits = logits / cfg.temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze()

            generated_ids = torch.cat([
                generated_ids,
                next_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)

            yield self.decode(generated_ids[0])

            if generated_ids.size(1) > self.model.config.max_seq_len:
                generated_ids = generated_ids[:, -self.model.config.max_seq_len:]

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            config: Generation config

        Returns:
            List of generated texts
        """
        return [self.generate(p, config) for p in prompts]

    def chat(
        self,
        message: str,
        conversation_history: Optional[List[str]] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate a response in a chat context.

        Args:
            message: User message
            conversation_history: Previous messages
            config: Generation config

        Returns:
            Model response
        """
        # Build context from history
        if conversation_history:
            context = " ".join(conversation_history[-5:]) + " " + message
        else:
            context = message

        return self.generate(context, config)


def quick_generate(
    model: ZeroBaseLLM,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0
) -> str:
    """
    Quick generation helper function.

    Args:
        model: ZeroBaseLLM model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature
    )
    generator = TextGenerator(model, config)
    return generator.generate(prompt)