"""
Normalization and residual connection modules.

Layer 8: Residual Connection + Layer Normalization

Implements:
- LayerNorm: Normalizes across features for training stability
- ResidualConnection: Adds input to sublayer output with dropout
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Normalizes across the feature dimension, helping with training stability.
    Unlike BatchNorm, LayerNorm works on a single sample independently.

    Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
    """

    def __init__(self, features: int, eps: float = 1e-6):
        """
        Initialize LayerNorm.

        Args:
            features: Number of features (dimension to normalize over)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

        # Learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (..., features)

        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and variance over the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    A simpler alternative to LayerNorm that only uses RMS normalization
    without mean centering. Faster and often works equally well.

    Formula: y = x / sqrt(mean(x^2) + eps) * gamma
    """

    def __init__(self, features: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.

        Args:
            features: Number of features
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


class ResidualConnection(nn.Module):
    """
    Residual Connection with Layer Normalization.

    Implements: output = LayerNorm(x + dropout(sublayer(x)))

    This is crucial for training deep networks as it allows gradients
    to flow directly through the skip connection.

    There are two common orderings:
    - Post-norm: x + sublayer(LayerNorm(x))  (original transformer)
    - Pre-norm: LayerNorm(x + sublayer(x))   (modern preference)

    We use pre-norm as it's more stable for training.
    """

    def __init__(
        self,
        features: int,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        """
        Initialize residual connection.

        Args:
            features: Number of features for LayerNorm
            dropout: Dropout rate for sublayer output
            pre_norm: If True, apply LayerNorm before residual addition
        """
        super().__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = pre_norm

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply residual connection.

        Args:
            x: Input tensor
            sublayer: Function to apply (e.g., attention, FFN)

        Returns:
            Output tensor with residual connection
        """
        if self.pre_norm:
            # Pre-norm: LayerNorm first, then sublayer, then add
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            # Post-norm: sublayer first, then add, then LayerNorm
            return self.norm(x + self.dropout(sublayer(x)))


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (Layer 7).

    FFN(x) = Linear(ReLU(Linear(x)))

    This is a simple two-layer MLP applied to each position independently.
    It provides the model with additional capacity to learn complex patterns.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize feed-forward network.

        Args:
            embed_dim: Input and output dimension
            ff_dim: Hidden dimension
            dropout: Dropout rate
            activation: Activation function ("relu", "gelu", "silu")
        """
        super().__init__()

        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)

        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GLU(nn.Module):
    """
    Gated Linear Unit variant of FFN.

    GLU(x) = (xW1 + b1) ⊗ σ(xW2 + b2)

    Used in some modern architectures (e.g., LLaMA) for improved performance.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize GLU.

        Args:
            embed_dim: Input and output dimension
            ff_dim: Hidden dimension (will be doubled for gate)
            dropout: Dropout rate
        """
        super().__init__()

        self.linear_gate = nn.Linear(embed_dim, ff_dim)
        self.linear_value = nn.Linear(embed_dim, ff_dim)
        self.linear_out = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.linear_gate.weight)
        nn.init.xavier_uniform_(self.linear_value.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GLU.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        gate = self.sigmoid(self.linear_gate(x))
        value = self.linear_value(x)
        x = gate * value
        x = self.dropout(x)
        x = self.linear_out(x)
        return x