"""
Zone C: Word Building Layers (9-12)

Layer 9: Character Clustering - detect word boundaries, pool characters
Layer 10: Word Embedding Projection - project to word-level vectors
Layer 11: Semantic Meaning Layer - cosine similarity for word proximity
Layer 12: Context Fusion - combine word vector + position + neighbor attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class CharacterClustering(nn.Module):
    """
    Layer 9: Character Clustering.

    Detects word boundaries using space and punctuation.
    Groups character embeddings within boundaries and applies pooling.

    Input: Character embeddings (batch, seq_len, embed_dim)
    Output: Word vectors (batch, num_words, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        pooling: str = "mean"
    ):
        """
        Initialize CharacterClustering.

        Args:
            embed_dim: Character embedding dimension
            pooling: Pooling method ("mean", "max", "first", "last")
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pooling = pooling

        # Word boundary characters (ASCII codes)
        # space=32, comma=44, period=46, exclamation=33, question=63
        self.boundary_ids = {32, 44, 46, 33, 63, 10, 13, 9}  # include newline, tab

    def detect_boundaries(
        self,
        char_ids: torch.Tensor
    ) -> List[List[Tuple[int, int]]]:
        """
        Detect word boundaries in character sequences.

        Args:
            char_ids: Character IDs (batch, seq_len)

        Returns:
            List of lists of (start, end) tuples for each batch
        """
        batch_size, seq_len = char_ids.shape
        all_boundaries = []

        for b in range(batch_size):
            boundaries = []
            word_start = None

            for i in range(seq_len):
                char_id = int(char_ids[b, i].item())

                if char_id in self.boundary_ids:
                    # Found boundary - end current word if any
                    if word_start is not None:
                        boundaries.append((word_start, i))
                        word_start = None
                else:
                    # Non-boundary character
                    if word_start is None:
                        word_start = i

            # Handle last word
            if word_start is not None:
                boundaries.append((word_start, seq_len))

            all_boundaries.append(boundaries)

        return all_boundaries

    def pool_characters(
        self,
        char_embeddings: torch.Tensor,
        boundaries: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Pool character embeddings into word vectors.

        Args:
            char_embeddings: Character embeddings (seq_len, embed_dim)
            boundaries: List of (start, end) tuples

        Returns:
            Word vectors (num_words, embed_dim)
        """
        if not boundaries:
            # No words found, return empty tensor
            return torch.zeros(0, self.embed_dim, device=char_embeddings.device)

        word_vectors = []
        for start, end in boundaries:
            if start >= end:
                continue

            word_chars = char_embeddings[start:end]

            if self.pooling == "mean":
                word_vec = word_chars.mean(dim=0)
            elif self.pooling == "max":
                word_vec = word_chars.max(dim=0)[0]
            elif self.pooling == "first":
                word_vec = word_chars[0]
            elif self.pooling == "last":
                word_vec = word_chars[-1]
            else:
                word_vec = word_chars.mean(dim=0)

            word_vectors.append(word_vec)

        if not word_vectors:
            return torch.zeros(0, self.embed_dim, device=char_embeddings.device)

        return torch.stack(word_vectors)

    def forward(
        self,
        char_embeddings: torch.Tensor,
        char_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]]]:
        """
        Cluster characters into words.

        Args:
            char_embeddings: Character embeddings (batch, seq_len, embed_dim)
            char_ids: Character IDs (batch, seq_len)

        Returns:
            Tuple of (word_vectors_per_batch, boundaries_per_batch)
        """
        batch_size = char_embeddings.size(0)
        device = char_embeddings.device

        # Detect boundaries for all batches
        all_boundaries = self.detect_boundaries(char_ids)

        # Pool characters for each batch
        all_word_vectors = []
        for b in range(batch_size):
            word_vecs = self.pool_characters(
                char_embeddings[b],
                all_boundaries[b]
            )
            all_word_vectors.append(word_vecs)

        return all_word_vectors, all_boundaries


class WordEmbeddingProjection(nn.Module):
    """
    Layer 10: Word Embedding Projection.

    Projects pooled character vectors to word-level embedding space.
    Linear(256 → 512) + activation
    """

    def __init__(
        self,
        char_embed_dim: int = 256,
        word_embed_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize WordEmbeddingProjection.

        Args:
            char_embed_dim: Input character embedding dimension
            word_embed_dim: Output word embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(char_embed_dim, word_embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to word embedding space.

        Args:
            x: Pooled character vectors (..., char_embed_dim)

        Returns:
            Word vectors (..., word_embed_dim)
        """
        return self.projection(x)


class SemanticMeaningLayer(nn.Module):
    """
    Layer 11: Semantic Meaning Layer.

    Builds semantic proximity using cosine similarity between word vectors.
    Words appearing in similar contexts develop similar vectors during training.

    This layer doesn't have learnable parameters - the semantic relationships
    emerge from the training process.
    """

    def __init__(self):
        """Initialize SemanticMeaningLayer."""
        super().__init__()

    def cosine_similarity(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute cosine similarity between tensors.

        Args:
            x1: First tensor
            x2: Second tensor
            dim: Dimension to compute similarity
            eps: Small constant for numerical stability

        Returns:
            Cosine similarity tensor
        """
        return F.cosine_similarity(x1, x2, dim=dim, eps=eps)

    def compute_similarity_matrix(
        self,
        word_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between words.

        Args:
            word_vectors: Word vectors (num_words, embed_dim)

        Returns:
            Similarity matrix (num_words, num_words)
        """
        # Normalize vectors
        normalized = F.normalize(word_vectors, p=2, dim=-1)

        # Compute similarity matrix
        similarity = torch.mm(normalized, normalized.t())

        return similarity

    def forward(
        self,
        word_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute semantic relationships.

        Args:
            word_vectors: Word vectors (num_words, embed_dim)

        Returns:
            Tuple of (word_vectors, similarity_matrix)
        """
        similarity_matrix = self.compute_similarity_matrix(word_vectors)
        return word_vectors, similarity_matrix


class ContextFusion(nn.Module):
    """
    Layer 12: Context Fusion.

    Combines word vector with positional context and neighboring word attention.
    Uses a small attention window of ±context_window words.

    This disambiguates polysemous words by considering their context.
    """

    def __init__(
        self,
        word_embed_dim: int = 512,
        context_window: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize ContextFusion.

        Args:
            word_embed_dim: Word embedding dimension
            context_window: Number of words on each side to consider
            num_heads: Number of attention heads for context attention
            dropout: Dropout rate
        """
        super().__init__()
        self.word_embed_dim = word_embed_dim
        self.context_window = context_window

        # Context attention mechanism
        self.context_attention = nn.MultiheadAttention(
            embed_dim=word_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(word_embed_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(word_embed_dim, word_embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

    def forward(
        self,
        word_vectors: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse context into word vectors.

        Args:
            word_vectors: Word vectors (batch, num_words, embed_dim)
                          or (num_words, embed_dim) for single sequence
            attention_mask: Optional attention mask

        Returns:
            Context-enriched word vectors
        """
        # Handle single sequence input
        if word_vectors.dim() == 2:
            word_vectors = word_vectors.unsqueeze(0)

        batch_size, num_words, embed_dim = word_vectors.shape

        # Apply self-attention for context fusion
        attn_out, _ = self.context_attention(
            word_vectors, word_vectors, word_vectors,
            key_padding_mask=attention_mask
        )

        # Residual connection and layer norm
        output = self.norm(word_vectors + attn_out)

        # Output projection
        output = self.output_proj(output)

        return output


class WordBuilder(nn.Module):
    """
    Zone C: Word Building (Layers 9-12).

    Complete pipeline for building word representations from characters.
    """

    def __init__(
        self,
        char_embed_dim: int = 256,
        word_embed_dim: int = 512,
        context_window: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize WordBuilder.

        Args:
            char_embed_dim: Character embedding dimension
            word_embed_dim: Word embedding dimension
            context_window: Context window size
            dropout: Dropout rate
        """
        super().__init__()

        # Layer 9: Character Clustering
        self.clustering = CharacterClustering(char_embed_dim)

        # Layer 10: Word Embedding Projection
        self.projection = WordEmbeddingProjection(
            char_embed_dim, word_embed_dim, dropout
        )

        # Layer 11: Semantic Meaning
        self.semantic = SemanticMeaningLayer()

        # Layer 12: Context Fusion
        self.context_fusion = ContextFusion(
            word_embed_dim, context_window, dropout=dropout
        )

    def forward(
        self,
        char_embeddings: torch.Tensor,
        char_ids: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[Tuple[int, int]]]]:
        """
        Build word representations from character embeddings.

        Args:
            char_embeddings: Character embeddings (batch, seq_len, char_embed_dim)
            char_ids: Character IDs (batch, seq_len)

        Returns:
            Tuple of (list of word vectors per batch, list of boundaries per batch)
        """
        # Layer 9: Cluster characters into words
        word_vectors_list, boundaries = self.clustering(char_embeddings, char_ids)

        # Process each batch
        processed_words = []
        for word_vecs in word_vectors_list:
            if word_vecs.size(0) == 0:
                processed_words.append(word_vecs)
                continue

            # Layer 10: Project to word embedding space
            word_vecs = self.projection(word_vecs)

            # Layer 11: Compute semantic relationships (no params, just similarity)
            word_vecs, _ = self.semantic(word_vecs)

            # Layer 12: Fuse context
            word_vecs = self.context_fusion(word_vecs)

            processed_words.append(word_vecs.squeeze(0) if word_vecs.dim() == 3 else word_vecs)

        return processed_words, boundaries