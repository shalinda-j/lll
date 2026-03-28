"""
Zone E: Output Layers (19-20)

Layer 19: Output Projection - project hidden state to vocab logits
Layer 20: Sampling/Generation - various sampling strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OutputProjection(nn.Module):
    """
    Layer 19: Output Projection.

    Projects final hidden state to logit scores over vocabulary.
    Linear(hidden_dim → vocab_size)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        vocab_size: int = 128
    ):
        """
        Initialize OutputProjection.

        Args:
            hidden_dim: Hidden state dimension
            vocab_size: Size of vocabulary
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.projection = nn.Linear(hidden_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Project hidden state to logits.

        Args:
            hidden: Hidden state (batch, seq_len, hidden_dim)
                    or (batch, hidden_dim) for single prediction

        Returns:
            Logits over vocabulary
        """
        return self.projection(hidden)


class Sampler(nn.Module):
    """
    Layer 20: Sampling/Generation.

    Implements various sampling strategies:
    - Greedy: argmax(logits)
    - Top-k: sample from top k most probable tokens
    - Nucleus (top-p): sample from tokens summing to p probability mass
    - Temperature: control randomness of sampling
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.9
    ):
        """
        Initialize Sampler.

        Args:
            temperature: Sampling temperature (higher = more random)
            top_k: Number of top tokens for top-k sampling
            top_p: Probability mass for nucleus sampling
        """
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Greedy sampling: select the most probable token.

        Args:
            logits: Logits (batch, vocab_size)

        Returns:
            Selected token IDs
        """
        return torch.argmax(logits, dim=-1)

    def top_k_sample(
        self,
        logits: torch.Tensor,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Top-k sampling: sample from top k most probable tokens.

        Args:
            logits: Logits (batch, vocab_size)
            k: Number of top tokens (default: self.top_k)

        Returns:
            Selected token IDs
        """
        k = k or self.top_k

        # Get top-k logits
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

        # Convert to probabilities
        probs = F.softmax(top_k_logits, dim=-1)

        # Sample from top-k
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Map back to vocabulary indices
        return top_k_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

    def nucleus_sample(
        self,
        logits: torch.Tensor,
        p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Nucleus (top-p) sampling: sample from tokens summing to p probability mass.

        Args:
            logits: Logits (batch, vocab_size)
            p: Probability threshold (default: self.top_p)

        Returns:
            Selected token IDs
        """
        p = p or self.top_p

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff index
        sorted_indices_to_remove = cumulative_probs > p
        # Shift to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Set removed logits to -inf
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        # Convert back to probabilities
        probs = F.softmax(sorted_logits, dim=-1)

        # Sample
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Map back to vocabulary indices
        return sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

    def temperature_sample(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Temperature sampling: adjust randomness of sampling.

        Args:
            logits: Logits (batch, vocab_size)
            temperature: Temperature value (default: self.temperature)

        Returns:
            Selected token IDs
        """
        temp = temperature or self.temperature

        if temp <= 0:
            return self.greedy(logits)

        # Apply temperature
        scaled_logits = logits / temp

        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Sample
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def sample(
        self,
        logits: torch.Tensor,
        strategy: str = "nucleus",
        **kwargs
    ) -> torch.Tensor:
        """
        Sample tokens using specified strategy.

        Args:
            logits: Logits (batch, vocab_size)
            strategy: Sampling strategy ("greedy", "top_k", "nucleus", "temperature")
            **kwargs: Additional arguments for specific strategies

        Returns:
            Selected token IDs
        """
        if strategy == "greedy":
            return self.greedy(logits)
        elif strategy == "top_k":
            return self.top_k_sample(logits, k=kwargs.get("k"))
        elif strategy == "nucleus":
            return self.nucleus_sample(logits, p=kwargs.get("p"))
        elif strategy == "temperature":
            return self.temperature_sample(logits, temperature=kwargs.get("temperature"))
        else:
            # Default to nucleus sampling
            return self.nucleus_sample(logits)


class OutputLayer(nn.Module):
    """
    Zone E: Output (Layers 19-20).

    Complete output layer combining projection and sampling.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        vocab_size: int = 128,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.9
    ):
        """
        Initialize OutputLayer.

        Args:
            hidden_dim: Hidden state dimension
            vocab_size: Size of vocabulary
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
        """
        super().__init__()

        # Layer 19: Output Projection
        self.projection = OutputProjection(hidden_dim, vocab_size)

        # Layer 20: Sampler
        self.sampler = Sampler(temperature, top_k, top_p)

    def forward(
        self,
        hidden: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get logits from hidden state.

        Args:
            hidden: Hidden state
            temperature: Optional temperature override

        Returns:
            Logits (scaled by temperature if provided)
        """
        logits = self.projection(hidden)

        if temperature is not None:
            logits = logits / temperature

        return logits

    def sample(
        self,
        hidden: torch.Tensor,
        strategy: str = "nucleus",
        **kwargs
    ) -> torch.Tensor:
        """
        Sample tokens from hidden state.

        Args:
            hidden: Hidden state
            strategy: Sampling strategy
            **kwargs: Additional arguments for sampler

        Returns:
            Sampled token IDs
        """
        logits = self.projection(hidden)
        return self.sampler.sample(logits, strategy, **kwargs)

    def generate(
        self,
        hidden: torch.Tensor,
        num_tokens: int = 1,
        strategy: str = "nucleus",
        **kwargs
    ) -> torch.Tensor:
        """
        Generate multiple tokens.

        Args:
            hidden: Hidden state (batch, seq_len, hidden_dim)
            num_tokens: Number of tokens to generate
            strategy: Sampling strategy
            **kwargs: Additional arguments

        Returns:
            Generated token IDs (batch, num_tokens)
        """
        # Use the last hidden state for generation
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]

        logits = self.projection(hidden)
        tokens = self.sampler.sample(logits, strategy, **kwargs)

        return tokens