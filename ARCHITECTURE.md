# Zero-Base LLM Architecture Diagram

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ZERO-BASE LLM: 22-LAYER ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   INPUT: "hello world"                                                          │
│      │                                                                          │
│      ▼                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │  ZONE A: FOUNDATION (Layers 0-3)                                            │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 0: Binary Foundation                                          │    │ │
│ │  │   "h" → [0,1,1,0,1,0,0,0]  (8 bits per character)                  │    │ │
│ │  ├─────────────────────────────────────────────────────────────────────┤    │ │
│ │  │ Layer 1: Byte Encoding                                              │    │ │
│ │  │   [0,1,1,0,1,0,0,0] → 104 (byte integer)                           │    │ │
│ │  ├─────────────────────────────────────────────────────────────────────┤    │ │
│ │  │ Layer 2: ASCII Mapping                                               │    │ │
│ │  │   104 → chr(104) = 'h' → ID=104                                    │    │ │
│ │  ├─────────────────────────────────────────────────────────────────────┤    │ │
│ │  │ Layer 3: Frequency Sorting                                           │    │ │
│ │  │   Track character frequencies for vocabulary optimization           │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │              input_ids: [104, 101, 108, 108, 111, 32, ...]                  │ │
│ │              Shape: (batch, seq_len) = (1, 11)                              │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                          │
│                                      ▼                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │  ZONE B: TRANSFORMER CORE (Layers 4-8)                                      │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 4: Character Embedding                                         │    │ │
│ │  │   input_ids → embedding_table → vectors                            │    │ │
│ │  │   + Sinusoidal Positional Encoding (sin/cos)                       │    │ │
│ │  │   Shape: (batch, seq_len, embed_dim) = (1, 11, 256)                │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layers 5-8: Transformer Block (repeated N=4 times)                  │    │ │
│ │  │  ┌─────────────────────────────────────────────────────────────┐    │    │ │
│ │  │  │ Layer 5-6: Multi-Head Attention (8 heads)                   │    │    │ │
│ │  │  │   Q = x @ W_Q, K = x @ W_K, V = x @ W_V                     │    │    │ │
│ │  │  │   Attention = softmax(QK^T / sqrt(d_k)) @ V                 │    │    │ │
│ │  │  │   8 parallel heads, each d_k=32, concatenated               │    │    │ │
│ │  │  └─────────────────────────────────────────────────────────────┘    │    │ │
│ │  │                              │                                        │    │ │
│ │  │                              ▼                                        │    │ │
│ │  │  ┌─────────────────────────────────────────────────────────────┐    │    │ │
│ │  │  │ Layer 7: Feed-Forward Network                               │    │    │ │
│ │  │  │   FFN(x) = Linear(GELU(Linear(x)))                          │    │    │ │
│ │  │  │   Hidden dim: 512 → Output: 256                             │    │    │ │
│ │  │  └─────────────────────────────────────────────────────────────┘    │    │ │
│ │  │                              │                                        │    │ │
│ │  │                              ▼                                        │    │ │
│ │  │  ┌─────────────────────────────────────────────────────────────┐    │    │ │
│ │  │  │ Layer 8: Residual + LayerNorm                               │    │    │ │
│ │  │  │   output = LayerNorm(x + sublayer(x))                       │    │    │ │
│ │  │  └─────────────────────────────────────────────────────────────┘    │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │              char_hidden: Shape (1, 11, 256)                                │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                          │
│                                      ▼                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │  ZONE C: WORD BUILDING (Layers 9-12)                                        │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 9: Character Clustering                                        │    │ │
│ │  │   Detect word boundaries: space(32), comma(44), period(46)...      │    │ │
│ │  │   "hello world" → boundaries: [(0,5), (6,11)]                       │    │ │
│ │  │   Pool characters → word vectors via mean pooling                   │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 10: Word Embedding Projection                                  │    │ │
│ │  │   Linear(256 → 512) + GELU                                          │    │ │
│ │  │   Shape: (num_words, 512) = (2, 512)                                │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 11: Semantic Meaning (No params)                               │    │ │
│ │  │   Compute cosine similarity between word vectors                    │    │ │
│ │  │   Similar words → similar vectors (emerges during training)         │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 12: Context Fusion                                             │    │ │
│ │  │   Self-attention over words (±3 word window)                        │    │ │
│ │  │   Disambiguates polysemous words via context                        │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │              word_vectors: Shape (2, 512)                                   │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                          │
│                                      ▼                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │  ZONE D: SENTENCE/PARAGRAPH (Layers 13-18)                                  │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 13: Word Sequence Attention                                    │    │ │
│ │  │   Transformer attention at word level                               │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 14: Grammar/Syntax Pattern                                     │    │ │
│ │  │   Learn syntactic templates (NP, VP patterns emerge)                │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 15: Phrase Construction                                        │    │ │
│ │  │   Merge adjacent words: "red apple" → single phrase vector          │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 16: Sentence Completion                                        │    │ │
│ │  │   P(next_word | context) = softmax(logits)                          │    │ │
│ │  │   Temperature controls randomness (T=0.7 focused, T=1.2 creative)   │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 17: Multi-Sentence Coherence                                   │    │ │
│ │  │   Maintain topic vector across sentences                            │    │ │
│ │  │   Penalize drift from topic (cosine similarity)                     │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 18: Paragraph Assembly                                         │    │ │
│ │  │   Structure: intro → body → closing                                 │    │ │
│ │  │   Stop when completeness_score > 0.85                               │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │              logits: Shape (num_words, vocab_size=128)                     │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                          │
│                                      ▼                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │  ZONE E: OUTPUT (Layers 19-20)                                              │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 19: Output Projection                                          │    │ │
│ │  │   Linear(hidden_dim → vocab_size)                                   │    │ │
│ │  │   hidden(512) → logits(128)                                         │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │  ┌─────────────────────────────────────────────────────────────────────┐    │ │
│ │  │ Layer 20: Sampling                                                   │    │ │
│ │  │   • Greedy: argmax(logits)                                          │    │ │
│ │  │   • Top-k: sample from top 10 tokens                                │    │ │
│ │  │   • Nucleus (top-p): sample from tokens with P=0.9 mass             │    │ │
│ │  └─────────────────────────────────────────────────────────────────────┘    │ │
│ │                              │                                               │ │
│ │                              ▼                                               │ │
│ │              next_token_id: Shape (1,) → 101 ('e')                         │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                          │
│                                      ▼                                          │
│                    OUTPUT: "hello world e..." (autoregressive)                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ╔═══════════════════════════════════════╗
                    ║   ZONE F: SELF-STUDY (Layers 21-22)  ║
                    ║   (Applied during training only)      ║
                    ╚═══════════════════════════════════════╝
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
        ▼                                                       ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│  Layer 21: Forward Self-Study │       │  Layer 22: Backward Self-Study│
│                               │       │                               │
│  After generating output:     │       │  After generating output:     │
│  • Predict next continuation  │       │  • Check consistency with     │
│  • Score quality of output    │       │    original input             │
│  • Use as reward signal       │       │  • Backprop consistency loss  │
│                               │       │                               │
│  Inspired by AZR self-play    │       │  Mirrors brain memory         │
│                               │       │  consolidation (MIT 2023)     │
└───────────────────────────────┘       └───────────────────────────────┘
        │                                               │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
                    L_total = L_task + α·L_forward + β·L_backward
                                │
                                ▼
                        Backpropagation
                    (Updates all transformer weights)
```

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SEED TEXTS (209 diverse English sentences)                               │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  SeedTextDataset                                                     │  │
│   │  • 209 sentences × 3 repetition = ~25,000 samples                   │  │
│   │  • Data augmentation (random shifts)                                │  │
│   │  • Sliding window: stride=1, seq_len=128                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Training Step                                                       │  │
│   │                                                                      │  │
│   │  1. Forward Pass:                                                    │  │
│   │     input_ids → Zone B → char_hidden → char_logits                  │  │
│   │                                                                      │  │
│   │  2. Loss Computation:                                                │  │
│   │     CrossEntropy(char_logits, targets)                              │  │
│   │     + self_study_loss (consistency check)                           │  │
│   │                                                                      │  │
│   │  3. Backward Pass:                                                   │  │
│   │     loss.backward()                                                  │  │
│   │     clip_grad_norm_(1.0)                                            │  │
│   │                                                                      │  │
│   │  4. Weight Update:                                                   │  │
│   │     optimizer.step()     (AdamW, lr=3e-4)                           │  │
│   │     scheduler.step()     (CosineAnnealing)                          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Overfitting Prevention                                              │  │
│   │  • Dropout: 0.2 in attention and FFN layers                         │  │
│   │  • Weight decay: 0.01                                               │  │
│   │  • Gradient clipping: 1.0                                           │  │
│   │  • Validation check: if loss < 1.0 but output random → warning      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Counts by Component

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PARAMETER COUNTS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SMALL CONFIG (embed_dim=128, num_blocks=2)                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Component                    │  Parameters                          │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │  Character Embedding          │  128 × 128 = 16,384                  │  │
│   │  Transformer Blocks (2×)      │  ~1,100,000                          │  │
│   │  Word Projection              │  128 × 256 = 32,768                  │  │
│   │  Context Fusion               │  ~200,000                            │  │
│   │  Sentence Builder             │  ~300,000                            │  │
│   │  Output Layers (2×)           │  ~70,000                             │  │
│   │  Self-Study System            │  ~150,000                            │  │
│   │  LayerNorms, etc.             │  ~400,000                            │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │  TOTAL                        │  ~2,276,871 parameters               │  │
│   │  Model Size                   │  8.81 MB                             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   MEDIUM CONFIG (embed_dim=256, num_blocks=4)                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  TOTAL                        │  ~7,954,951 parameters               │  │
│   │  Model Size                   │  30.85 MB                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   LARGE CONFIG (embed_dim=512, num_blocks=8)                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  TOTAL                        │  ~28,891,143 parameters              │  │
│   │  Model Size                   │  112.21 MB                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   All configs are well under the 200MB target!                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tensor Shape Flow

```
Input Text: "hello world"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 0-3: Tokenization                                      │
│   "hello world" → [104, 101, 108, 108, 111, 32, 119, ...]   │
│   Shape: (seq_len,) = (11,)                                  │
│   After batching: (batch, seq_len) = (1, 11)                │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Character Embedding                                  │
│   Shape: (1, 11, 128) [small] or (1, 11, 256) [medium]      │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Layers 5-8: Transformer Blocks (×N)                          │
│   Each block maintains: (batch, seq_len, embed_dim)          │
│   Attention weights: (batch, num_heads, seq_len, seq_len)    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 9-12: Word Building                                     │
│   Input: (1, 11, 128)                                        │
│   Output: List of word vectors per batch                     │
│   Word vectors: (num_words, word_embed_dim) = (2, 256)       │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Layers 13-18: Sentence Building                               │
│   Input: (2, 256)                                            │
│   Output: logits over vocabulary                             │
│   Shape: (num_words, vocab_size) = (2, 128)                  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Layers 19-20: Output                                          │
│   Final logits: (batch, seq_len, vocab_size)                 │
│   For training: (1, 10, 128)  [seq_len-1 due to shift]       │
│   For generation: next token sampled from logits[-1]         │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Generated Token: 101 ('e')
```

---

## Self-Study Mechanism (Layers 21-22)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BIDIRECTIONAL SELF-STUDY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FORWARD SELF-STUDY (Layer 21)                                            │
│   ══════════════════════════════                                            │
│                                                                             │
│   Purpose: Predict what comes NEXT after generation                        │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  Generated Output: "hello world"                                   │    │
│   │       │                                                            │    │
│   │       ▼                                                            │    │
│   │  hidden_state (last position)                                      │    │
│   │       │                                                            │    │
│   │       ▼                                                            │    │
│   │  continuation_head → predict next tokens                           │    │
│   │       │                                                            │    │
│   │       ▼                                                            │    │
│   │  quality_scorer → score the quality of prediction                  │    │
│   │       │                                                            │    │
│   │       ▼                                                            │    │
│   │  Reward signal for weight update                                   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   BACKWARD SELF-STUDY (Layer 22)                                           │
│   ═══════════════════════════════                                          │
│                                                                             │
│   Purpose: Check consistency with INPUT context                           │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  Original Input: "hello"                                           │    │
│   │  Generated Output: "hello world"                                   │    │
│   │       │                                                            │    │
│   │       ▼                                                            │    │
│   │  encode_context(input) + encode_context(output)                    │    │
│   │       │                                                            │    │
│   │       ▼                                                            │    │
│   │  consistency_network → score (0-1)                                 │    │
│   │       │                                                            │    │
│   │       ▼                                                            │    │
│   │  consistency_loss = 1.0 - consistency_score                        │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   COMBINED LOSS:                                                           │
│   ═══════════════                                                          │
│                                                                             │
│   L_total = L_task + 0.3 × L_forward + 0.3 × L_backward                   │
│                                                                             │
│   Where:                                                                   │
│     L_task = CrossEntropyLoss(predictions, targets)                       │
│     L_forward = Forward study quality loss                                │
│     L_backward = 1.0 - consistency_score                                   │
│                                                                             │
│   This mirrors how human brain consolidates memory:                        │
│   Synaptic weights encode experience, not data files.                      │
│   (Source: MIT Neuroscience 2023)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Memory Design (Brain-Inspired)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WHAT THE MODEL STORES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ❌ NOT STORED (No data files):                                           │
│   ─────────────────────────────────                                        │
│   • Raw training text                                                      │
│   • Database of facts                                                      │
│   • Lookup tables for answers                                              │
│   • Cached embeddings                                                      │
│                                                                             │
│   ✅ STORED IN WEIGHTS ONLY (Synaptic patterns):                           │
│   ─────────────────────────────────────────                                │
│   • Pattern frequencies (Layer 3)                                          │
│   • Character co-occurrence patterns (Layers 5-6)                          │
│   • Word-level semantic relationships (Layer 11)                           │
│   • Syntactic templates (Layer 14)                                         │
│   • Output quality heuristics (Layers 21-22)                               │
│                                                                             │
│   This is analogous to how the human brain stores memory:                   │
│   • Neurons don't store text files                                         │
│   • Memory = strength of synaptic connections                              │
│   • Learning = changing synaptic weights                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Configurations

| Config | embed_dim | num_heads | num_blocks | word_embed_dim | Parameters | Size |
|--------|-----------|-----------|------------|----------------|------------|------|
| Small | 128 | 4 | 2 | 256 | 2.3M | 8.8 MB |
| Medium | 256 | 8 | 4 | 512 | 8.0M | 30.9 MB |
| Large | 512 | 8 | 8 | 768 | 28.9M | 112.2 MB |

---

## Usage

```bash
# Interactive mode
python run.py --interactive

# Single generation
python run.py --prompt "hello world" --max-tokens 50

# Training with self-study
python run.py --train --steps 10000 --config small

# Different model sizes
python run.py --config medium --prompt "test"
python run.py --config large --prompt "test"
```