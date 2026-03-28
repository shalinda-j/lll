"""
Zone D: Sentence & Paragraph Building Layers (13-18)

Layer 13: Word Sequence Attention - transformer attention at word level
Layer 14: Grammar/Syntax Pattern Layer - emergent syntactic templates
Layer 15: Phrase Construction Layer - merge words into phrases
Layer 16: Sentence Completion Predictor - autoregressive prediction
Layer 17: Multi-Sentence Coherence Layer - topic vector maintenance
Layer 18: Paragraph/Description Assembly - intro→body→closing structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class WordSequenceAttention(nn.Module):
    """
    Layer 13: Word Sequence Attention.

    Applies transformer attention at the word level.
    Learns word order: "dog bites man" ≠ "man bites dog"
    """

    def __init__(
        self,
        word_embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        """
        Initialize WordSequenceAttention.

        Args:
            word_embed_dim: Word embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.word_embed_dim = word_embed_dim

        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=word_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(word_embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim, word_embed_dim),
            nn.Dropout(p=dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(word_embed_dim)
        self.norm2 = nn.LayerNorm(word_embed_dim)

    def forward(
        self,
        word_vectors: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply word-level attention.

        Args:
            word_vectors: Word vectors (batch, num_words, embed_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention(
            word_vectors, word_vectors, word_vectors,
            key_padding_mask=mask,
            need_weights=True
        )
        x = self.norm1(word_vectors + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


class GrammarSyntaxLayer(nn.Module):
    """
    Layer 14: Grammar/Syntax Pattern Layer.

    Learns syntactic templates through self-study.
    Identifies noun phrases, verb phrases via positional patterns.
    No hard-coded rules - patterns emerge from attention weights.
    """

    def __init__(
        self,
        word_embed_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize GrammarSyntaxLayer.

        Args:
            word_embed_dim: Word embedding dimension
            hidden_dim: Hidden dimension for pattern detection
            dropout: Dropout rate
        """
        super().__init__()

        # Position-aware pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Linear(word_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, word_embed_dim),
            nn.Dropout(p=dropout)
        )

        # Position encoding for syntax patterns
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 512, word_embed_dim) * 0.02
        )

        self.norm = nn.LayerNorm(word_embed_dim)

    def forward(
        self,
        word_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply syntax pattern detection.

        Args:
            word_vectors: Word vectors (batch, num_words, embed_dim)

        Returns:
            Syntax-enriched word vectors
        """
        num_words = word_vectors.size(1)

        # Add position encoding
        pos_enc = self.pos_encoding[:, :num_words, :]
        x = word_vectors + pos_enc

        # Detect patterns
        patterns = self.pattern_detector(x)

        # Residual connection
        return self.norm(word_vectors + patterns)


class PhraseConstruction(nn.Module):
    """
    Layer 15: Phrase Construction Layer.

    Merges word vectors into phrase-level representations.
    Example: "red apple" → single composite semantic unit.
    """

    def __init__(
        self,
        word_embed_dim: int = 512,
        phrase_embed_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize PhraseConstruction.

        Args:
            word_embed_dim: Word embedding dimension
            phrase_embed_dim: Phrase embedding dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Phrase merge function
        self.merge_function = nn.Sequential(
            nn.Linear(word_embed_dim * 2, phrase_embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(phrase_embed_dim, phrase_embed_dim)
        )

        # Gating mechanism for deciding whether to merge
        self.merge_gate = nn.Sequential(
            nn.Linear(word_embed_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        word_vectors: torch.Tensor,
        merge_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Construct phrases from adjacent words.

        Args:
            word_vectors: Word vectors (batch, num_words, embed_dim)
            merge_scores: Optional pre-computed merge scores

        Returns:
            Phrase vectors (batch, num_phrases, embed_dim)
        """
        if word_vectors.dim() == 2:
            word_vectors = word_vectors.unsqueeze(0)

        batch_size, num_words, embed_dim = word_vectors.shape

        if num_words < 2:
            return word_vectors

        # Compute merge scores between adjacent words
        phrase_vectors = []
        i = 0
        while i < num_words:
            if i < num_words - 1:
                # Consider merging with next word
                concat = torch.cat([
                    word_vectors[:, i, :],
                    word_vectors[:, i+1, :]
                ], dim=-1)

                gate_value = self.merge_gate(concat).squeeze(-1)

                if gate_value.mean() > 0.5:
                    # Merge words into phrase
                    merged = self.merge_function(concat)
                    phrase_vectors.append(merged)
                    i += 2
                else:
                    phrase_vectors.append(word_vectors[:, i, :])
                    i += 1
            else:
                phrase_vectors.append(word_vectors[:, i, :])
                i += 1

        if not phrase_vectors:
            return word_vectors

        # Stack phrase vectors
        result = torch.stack(phrase_vectors, dim=1)
        return result


class SentenceCompletion(nn.Module):
    """
    Layer 16: Sentence Completion Predictor.

    Core autoregressive prediction:
    P(word_{n+1} | word_1...word_n) = softmax(Linear(h_n))

    Temperature parameter T controls randomness:
    - T=0.7 for focused output
    - T=1.2 for creative output
    """

    def __init__(
        self,
        word_embed_dim: int = 512,
        vocab_size: int = 128,
        temperature: float = 1.0
    ):
        """
        Initialize SentenceCompletion.

        Args:
            word_embed_dim: Word embedding dimension
            vocab_size: Size of vocabulary
            temperature: Sampling temperature
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature

        # Output projection to vocabulary
        self.output_proj = nn.Linear(word_embed_dim, vocab_size)

    def forward(
        self,
        hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next word probabilities.

        Args:
            hidden: Hidden state (batch, seq_len, embed_dim)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        logits = self.output_proj(hidden) / self.temperature
        return logits

    def predict_next(
        self,
        hidden: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Predict next token distribution.

        Args:
            hidden: Hidden state (batch, embed_dim)
            temperature: Optional temperature override

        Returns:
            Probability distribution (batch, vocab_size)
        """
        temp = temperature if temperature is not None else self.temperature
        logits = self.output_proj(hidden) / temp
        return F.softmax(logits, dim=-1)


class MultiSentenceCoherence(nn.Module):
    """
    Layer 17: Multi-Sentence Coherence Layer.

    Maintains topic vector across sentences.
    Penalizes output that drifts from topic.
    Uses cosine similarity between current sentence and topic vector.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        topic_dim: int = 256,
        coherence_weight: float = 0.5
    ):
        """
        Initialize MultiSentenceCoherence.

        Args:
            embed_dim: Sentence embedding dimension
            topic_dim: Topic vector dimension
            coherence_weight: Weight for coherence in loss
        """
        super().__init__()
        self.coherence_weight = coherence_weight

        # Topic projection
        self.topic_proj = nn.Linear(embed_dim, topic_dim)

        # Topic memory (updated incrementally)
        self.register_buffer('topic_vector', torch.zeros(1, topic_dim))

    def update_topic(self, sentence_vector: torch.Tensor) -> None:
        """
        Update topic vector with new sentence.

        Args:
            sentence_vector: Sentence representation
        """
        new_topic = self.topic_proj(sentence_vector)
        # Exponential moving average update
        self.topic_vector = 0.9 * self.topic_vector + 0.1 * new_topic.detach()

    def compute_coherence_score(
        self,
        sentence_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coherence with current topic.

        Args:
            sentence_vector: Sentence representation

        Returns:
            Coherence score (0-1)
        """
        projected = self.topic_proj(sentence_vector)
        coherence = F.cosine_similarity(
            projected,
            self.topic_vector.expand(projected.size(0), -1),
            dim=-1
        )
        return coherence

    def forward(
        self,
        sentence_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply coherence tracking.

        Args:
            sentence_vectors: Sentence vectors (batch, num_sentences, embed_dim)

        Returns:
            Tuple of (output, coherence_scores)
        """
        # Compute coherence for each sentence
        batch_size, num_sentences, embed_dim = sentence_vectors.shape
        coherence_scores = []

        for i in range(num_sentences):
            score = self.compute_coherence_score(sentence_vectors[:, i, :])
            coherence_scores.append(score)

        coherence_scores = torch.stack(coherence_scores, dim=1)

        return sentence_vectors, coherence_scores


class ParagraphAssembler(nn.Module):
    """
    Layer 18: Paragraph/Description Assembly.

    Structures output into:
    - Introduction sentence
    - Body sentences
    - Closing sentence

    Tracks sentence count and semantic completeness score.
    Stops generation when completeness_score > threshold.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        max_sentences: int = 10,
        completeness_threshold: float = 0.85
    ):
        """
        Initialize ParagraphAssembler.

        Args:
            embed_dim: Sentence embedding dimension
            max_sentences: Maximum sentences per paragraph
            completeness_threshold: Threshold to stop generation
        """
        super().__init__()
        self.max_sentences = max_sentences
        self.completeness_threshold = completeness_threshold

        # Completeness estimator
        self.completeness_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # Sentence type classifier (intro/body/closing)
        self.sentence_type_classifier = nn.Linear(embed_dim, 3)

    def forward(
        self,
        sentence_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assemble paragraph from sentences.

        Args:
            sentence_vectors: Sentence vectors (batch, num_sentences, embed_dim)

        Returns:
            Tuple of (output, completeness_scores, sentence_types)
        """
        # Estimate completeness
        completeness_scores = self.completeness_estimator(sentence_vectors).squeeze(-1)

        # Classify sentence types
        sentence_types = self.sentence_type_classifier(sentence_vectors)
        sentence_types = F.softmax(sentence_types, dim=-1)

        return sentence_vectors, completeness_scores, sentence_types

    def should_stop(
        self,
        completeness_score: torch.Tensor
    ) -> bool:
        """
        Determine if paragraph is complete.

        Args:
            completeness_score: Current completeness score

        Returns:
            True if generation should stop
        """
        return completeness_score.mean().item() > self.completeness_threshold


class SentenceBuilder(nn.Module):
    """
    Zone D: Sentence & Paragraph Building (Layers 13-18).

    Complete pipeline for building sentences and paragraphs from words.
    """

    def __init__(
        self,
        word_embed_dim: int = 512,
        vocab_size: int = 128,
        max_sentences: int = 10,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        """
        Initialize SentenceBuilder.

        Args:
            word_embed_dim: Word embedding dimension
            vocab_size: Size of vocabulary
            max_sentences: Maximum sentences per paragraph
            temperature: Sampling temperature
            dropout: Dropout rate
        """
        super().__init__()

        # Layer 13: Word Sequence Attention
        self.word_attention = WordSequenceAttention(
            word_embed_dim, num_heads=8, dropout=dropout
        )

        # Layer 14: Grammar/Syntax
        self.grammar = GrammarSyntaxLayer(word_embed_dim, dropout=dropout)

        # Layer 15: Phrase Construction (output dimension matches word_embed_dim)
        self.phrase = PhraseConstruction(word_embed_dim, phrase_embed_dim=word_embed_dim, dropout=dropout)

        # Layer 16: Sentence Completion
        self.completion = SentenceCompletion(
            word_embed_dim, vocab_size, temperature
        )

        # Layer 17: Multi-Sentence Coherence
        self.coherence = MultiSentenceCoherence(word_embed_dim)

        # Layer 18: Paragraph Assembly
        self.paragraph = ParagraphAssembler(
            word_embed_dim, max_sentences
        )

    def forward(
        self,
        word_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build sentence/paragraph from word vectors.

        Args:
            word_vectors: Word vectors (batch, num_words, embed_dim)

        Returns:
            Tuple of (logits, coherence_scores)
        """
        # Handle single sequence
        if word_vectors.dim() == 2:
            word_vectors = word_vectors.unsqueeze(0)

        # Layer 13: Word attention
        x, _ = self.word_attention(word_vectors)

        # Layer 14: Grammar patterns
        x = self.grammar(x)

        # Layer 15: Phrase construction
        x = self.phrase(x)

        # Layer 16: Sentence completion (get logits)
        logits = self.completion(x)

        # Compute coherence (for training signal)
        # Mean pool across words to get sentence vector
        sentence_vec = x.mean(dim=1, keepdim=True)
        _, coherence_scores = self.coherence(sentence_vec)

        return logits, coherence_scores