"""
Working Diagram: Zero-Base LLM Logical Flow
============================================

This diagram shows the ACTUAL working pattern - how data flows,
how operations execute, and how training/generation work.

Run this file to see a visual demonstration of the model's logic.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from zero_base_llm import ZeroBaseConfig, ZeroBaseLLM
from zero_base_llm.tokenizer import BinaryTokenizer


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


def show_tensor(name, tensor):
    print(f"    {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")


def working_diagram():
    """Show the complete working pattern of the Zero-Base LLM."""

    # ==================== INITIALIZATION ====================
    print_section("PHASE 1: INITIALIZATION")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     MODEL INITIALIZATION                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   config = ZeroBaseConfig.small()                                  │
    │       │                                                             │
    │       ├── vocab_size = 128 (ASCII characters)                      │
    │       ├── embed_dim = 128                                          │
    │       ├── num_heads = 4                                            │
    │       ├── num_blocks = 2                                           │
    │       └── word_embed_dim = 256                                     │
    │                                                                     │
    │   model = ZeroBaseLLM(config)                                      │
    │       │                                                             │
    │       ├── Zone A: BinaryFoundation (tokenizer)                     │
    │       ├── Zone B: TransformerCore (2 blocks)                       │
    │       ├── Zone C: WordBuilder                                      │
    │       ├── Zone D: SentenceBuilder                                  │
    │       ├── Zone E: OutputLayer (char + word)                        │
    │       └── Zone F: SelfStudySystem                                  │
    │                                                                     │
    │   Total Parameters: ~2.3M | Size: ~9 MB                            │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    config = ZeroBaseConfig.small()
    model = ZeroBaseLLM(config)
    tokenizer = BinaryTokenizer()

    print(f"\n    ✓ Model initialized with {model.count_parameters():,} parameters")
    print(f"    ✓ Model size: {model.get_model_size_mb():.2f} MB")

    # ==================== ZONE A: TOKENIZATION ====================
    print_section("PHASE 2: ZONE A - TOKENIZATION (Layers 0-3)")

    input_text = "hello world"
    print(f"\n    Input Text: '{input_text}'")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 0: Binary Foundation                                        │
    │  ─────────────────────────                                         │
    │  Each character → 8 bits                                           │
    │                                                                     │
    │  'h' (ASCII 104) → [0,1,1,0,1,0,0,0]                              │
    │  'e' (ASCII 101) → [0,1,1,0,0,1,0,1]                              │
    │  'l' (ASCII 108) → [0,1,1,0,1,1,0,0]                              │
    │  ...                                                                │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 1: Byte Encoding                                            │
    │  ─────────────────────────                                         │
    │  8 bits → 1 byte integer (0-255)                                   │
    │                                                                     │
    │  [0,1,1,0,1,0,0,0] → 104                                          │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 2: ASCII Mapping                                            │
    │  ─────────────────────────                                         │
    │  byte → ASCII character ID (0-127)                                 │
    │                                                                     │
    │  104 → 'h' → char_id = 104                                        │
    │  101 → 'e' → char_id = 101                                        │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 3: Frequency Tracking                                       │
    │  ─────────────────────────                                         │
    │  Track character frequencies for vocabulary optimization           │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    # Show actual tokenization
    input_ids = tokenizer.encode(input_text)
    print(f"\n    Encoded: {input_text}")
    print(f"    → input_ids = {input_ids.tolist()}")

    # Show character breakdown
    print("\n    Character breakdown:")
    for i, char in enumerate(input_text):
        bits = tokenizer.char_to_bits(char)
        print(f"      '{char}' → bits: {bits} → byte: {ord(char)} → id: {input_ids[i].item()}")

    input_ids = input_ids.unsqueeze(0)  # Add batch dimension
    print(f"\n    After batching: input_ids.shape = {list(input_ids.shape)}")

    # ==================== ZONE B: TRANSFORMER CORE ====================
    print_section("PHASE 3: ZONE B - TRANSFORMER CORE (Layers 4-8)")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 4: Character Embedding                                      │
    │  ──────────────────────────────────                                │
    │                                                                     │
    │   input_ids (batch, seq_len)                                       │
    │         │                                                           │
    │         ▼                                                           │
    │   ┌─────────────────┐                                              │
    │   │ Embedding Table │  (128 chars × 128 dim)                       │
    │   │   char_id → vec │                                              │
    │   └─────────────────┘                                              │
    │         │                                                           │
    │         ▼                                                           │
    │   + Positional Encoding (sin/cos)                                  │
    │         │                                                           │
    │         ▼                                                           │
    │   char_embeds: (batch, seq_len, embed_dim)                         │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layers 5-8: Transformer Block × N (repeated)                      │
    │  ─────────────────────────────────────────                         │
    │                                                                     │
    │   For each block:                                                   │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Layer 5-6: Multi-Head Attention                            │  │
    │   │                                                             │  │
    │   │    Q = x @ W_Q    (query)                                   │  │
    │   │    K = x @ W_K    (key)                                     │  │
    │   │    V = x @ W_V    (value)                                   │  │
    │   │                                                             │  │
    │   │    Split into 4 heads (embed_dim/4 = 32 each)              │  │
    │   │                                                             │  │
    │   │    Attention = softmax(QK^T / sqrt(d_k)) @ V               │  │
    │   │                                                             │  │
    │   │    Concatenate heads → Linear → Output                     │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                         │                                           │
    │                         ▼                                           │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Layer 7: Feed-Forward Network                              │  │
    │   │                                                             │  │
    │   │    FFN(x) = Linear(GELU(Linear(x)))                        │  │
    │   │                                                             │  │
    │   │    x (128) → Linear → (256) → GELU → Linear → (128)       │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                         │                                           │
    │                         ▼                                           │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Layer 8: Residual + LayerNorm                              │  │
    │   │                                                             │  │
    │   │    output = LayerNorm(x + sublayer_output)                 │  │
    │   │                                                             │  │
    │   │    This allows gradients to flow through skip connections  │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    with torch.no_grad():
        char_hidden, attention_weights = model.transformer_core(input_ids)

    print(f"\n    char_hidden.shape = {list(char_hidden.shape)}")
    print(f"    attention_weights: {len(attention_weights)} blocks")
    print(f"    Each attention: {list(attention_weights[0].shape)}")

    # ==================== ZONE C: WORD BUILDING ====================
    print_section("PHASE 4: ZONE C - WORD BUILDING (Layers 9-12)")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 9: Character Clustering                                     │
    │  ──────────────────────────────                                    │
    │                                                                     │
    │   Detect word boundaries using:                                    │
    │   • Space (ASCII 32)                                               │
    │   • Comma (ASCII 44)                                               │
    │   • Period (ASCII 46)                                              │
    │   • !? (ASCII 33, 63)                                              │
    │                                                                     │
    │   "hello world" → boundaries: [(0,5), (6,11)]                     │
    │                                                                     │
    │   Pool characters within each word:                                │
    │   'h','e','l','l','o' → mean([vec_h, vec_e, ...]) = word_vec     │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 10: Word Embedding Projection                               │
    │  ──────────────────────────────────────                            │
    │                                                                     │
    │   Linear(128 → 256) + GELU                                         │
    │   char_vec → richer word_vec                                       │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 11: Semantic Meaning (No learnable params)                  │
    │  ─────────────────────────────────────────────                     │
    │                                                                     │
    │   Compute cosine similarity between word vectors:                  │
    │   similarity(w1, w2) = dot(w1, w2) / (||w1|| × ||w2||)            │
    │                                                                     │
    │   Words in similar contexts → similar vectors (during training)   │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 12: Context Fusion                                          │
    │  ───────────────────────                                           │
    │                                                                     │
    │   Self-attention over words (window ±3)                            │
    │   Disambiguates words like "bank" (river vs money)                 │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    word_vectors_list, word_boundaries = model.word_builder(char_hidden, input_ids)

    print(f"\n    Word boundaries detected: {word_boundaries}")
    print(f"    Number of words: {len(word_vectors_list[0])}")
    if len(word_vectors_list[0]) > 0:
        print(f"    Word vector shape: {list(word_vectors_list[0].shape)}")

    # ==================== ZONE D: SENTENCE BUILDING ====================
    print_section("PHASE 5: ZONE D - SENTENCE/PARAGRAPH (Layers 13-18)")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 13: Word Sequence Attention                                 │
    │  ─────────────────────────────────                                 │
    │   Attention at word level: "dog bites man" ≠ "man bites dog"      │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 14: Grammar/Syntax Pattern                                  │
    │  ───────────────────────────────                                   │
    │   Learn syntactic templates through self-study                     │
    │   Noun phrases, verb patterns emerge from attention weights        │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 15: Phrase Construction                                     │
    │  ───────────────────────────                                       │
    │   Merge adjacent words:                                            │
    │   "red" + "apple" → "red apple" (single phrase vector)            │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 16: Sentence Completion                                     │
    │  ────────────────────────────                                      │
    │   P(next_word | context) = softmax(logits / temperature)          │
    │                                                                     │
    │   Temperature control:                                             │
    │   • T = 0.7 → focused, deterministic                              │
    │   • T = 1.0 → normal                                               │
    │   • T = 1.2 → creative, random                                     │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 17: Multi-Sentence Coherence                                │
    │  ─────────────────────────────────                                 │
    │   Maintain topic vector across sentences                           │
    │   Penalize drift: cosine_sim(current, topic)                       │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 18: Paragraph Assembly                                      │
    │  ─────────────────────────                                         │
    │   Structure: Introduction → Body → Closing                         │
    │   Stop when completeness_score > 0.85                              │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    # ==================== ZONE E: OUTPUT ====================
    print_section("PHASE 6: ZONE E - OUTPUT (Layers 19-20)")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 19: Output Projection                                       │
    │  ─────────────────────────                                         │
    │                                                                     │
    │   Linear(hidden_dim → vocab_size)                                  │
    │   hidden (256/512) → logits (128)                                  │
    │                                                                     │
    │   Each position gets 128 logits (one per ASCII character)          │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Layer 20: Sampling Strategies                                     │
    │  ─────────────────────────────                                     │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  GREEDY: argmax(logits)                                     │  │
    │   │  → Always pick highest probability token                    │  │
    │   │  → Most deterministic                                       │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  TOP-K: Sample from top k tokens                            │  │
    │   │  → Keep diversity limited                                   │  │
    │   │  → k=10 is default                                          │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  NUCLEUS (Top-P): Sample from tokens with cumulative P      │  │
    │   │  → Sort by probability, keep tokens until sum ≥ p           │  │
    │   │  → p=0.9 is default (90% probability mass)                  │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    with torch.no_grad():
        outputs = model(input_ids, use_self_study=False)
        logits = outputs["logits"]

    print(f"\n    logits.shape = {list(logits.shape)}")
    print(f"    logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

    # Show probability distribution
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_probs, top_indices = torch.topk(probs, 5)
    print(f"\n    Top 5 predicted next characters:")
    for i in range(5):
        char = chr(top_indices[i].item()) if top_indices[i].item() >= 32 else '?'
        print(f"      '{char}' (id={top_indices[i].item()}) : {top_probs[i].item():.4f}")

    # ==================== GENERATION LOOP ====================
    print_section("PHASE 7: AUTOGRESSIVE GENERATION")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    GENERATION LOOP                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   input_ids = encode("hello")                                      │
    │                                                                     │
    │   FOR each new token (up to max_tokens):                           │
    │       │                                                             │
    │       ▼                                                             │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Forward pass through all zones                              │  │
    │   │  input_ids → Zone B → ... → Zone E → logits                 │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │       │                                                             │
    │       ▼                                                             │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Sample next token from logits[-1]                          │  │
    │   │  next_token = sampler.sample(logits, strategy="nucleus")    │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │       │                                                             │
    │       ▼                                                             │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Append to input_ids                                        │  │
    │   │  input_ids = [input_ids, next_token]                        │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │       │                                                             │
    │       ▼                                                             │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Truncate if too long (max_seq_len)                         │  │
    │   │  if len > max_seq_len: truncate from start                  │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │       │                                                             │
    │       └───────────────┐                                             │
    │                       │                                             │
    │                       ▼                                             │
    │   NEXT ITERATION or STOP                                            │
    │                                                                     │
    │   Output: decode(input_ids) → "hello world..."                     │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    # Generate text
    generated = model.generate("hello", max_new_tokens=20, temperature=0.8)
    print(f"\n    Input: 'hello'")
    print(f"    Generated: '{generated}'")

    # ==================== ZONE F: SELF-STUDY ====================
    print_section("PHASE 8: ZONE F - SELF-STUDY TRAINING (Layers 21-22)")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                TRAINING PIPELINE                                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   1. FORWARD PASS (Layer 21 - Forward Self-Study)                  │
    │   ─────────────────────────────────────────────                     │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Input: seed_text = "hello world how are you"               │  │
    │   │                                                             │  │
    │   │  Create training pairs (autoregressive):                    │  │
    │   │    input:  "hello world how are yo"                         │  │
    │   │    target: "ello world how are you"                         │  │
    │   │                                                             │  │
    │   │  Forward: input → model → logits                           │  │
    │   │                                                             │  │
    │   │  Compute Loss:                                              │  │
    │   │    L_task = CrossEntropyLoss(logits, targets)              │  │
    │   │                                                             │  │
    │   │  Predict continuation:                                      │  │
    │   │    continuation_logits = forward_study(hidden_state)        │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    │   2. BACKWARD PASS (Layer 22 - Backward Self-Study)                │
    │   ──────────────────────────────────────────────────                │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Check consistency:                                         │  │
    │   │    input_hidden = encode(original_input)                    │  │
    │   │    output_hidden = encode(generated_output)                 │  │
    │   │                                                             │  │
    │   │  consistency_score = consistency_net(input, output)         │  │
    │   │  L_backward = 1.0 - consistency_score                       │  │
    │   │                                                             │  │
    │   │  This ensures output is coherent with input                 │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    │   3. WEIGHT UPDATE                                                 │
    │   ──────────────────                                               │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  Combined Loss:                                              │  │
    │   │    L_total = L_task + 0.3×L_forward + 0.3×L_backward        │  │
    │   │                                                             │  │
    │   │  Backpropagation:                                            │  │
    │   │    1. L_total.backward()      # Compute gradients           │  │
    │   │    2. clip_grad_norm_(1.0)    # Prevent exploding grads     │  │
    │   │    3. optimizer.step()        # Update weights              │  │
    │   │    4. optimizer.zero_grad()   # Reset for next step         │  │
    │   │                                                             │  │
    │   │  Optimizer: AdamW (lr=3e-4, weight_decay=0.01)              │  │
    │   │  Scheduler: CosineAnnealing (gradually reduce lr)           │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    │   4. OVERFITTING PREVENTION                                        │
    │   ──────────────────────────                                        │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │  • Dropout: 0.2 in attention and FFN layers                 │  │
    │   │  • Weight decay: 0.01 (L2 regularization)                   │  │
    │   │  • Gradient clipping: max norm = 1.0                        │  │
    │   │  • 209 diverse seed sentences                               │  │
    │   │  • Data augmentation (random shifts)                        │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    # Show training step
    model.train()
    targets = input_ids[:, 1:]
    inputs = input_ids[:, :-1]

    with torch.no_grad():
        outputs = model(inputs, targets=targets, use_self_study=False)

    print(f"\n    Training example:")
    print(f"      Input shape: {list(inputs.shape)}")
    print(f"      Target shape: {list(targets.shape)}")
    print(f"      Loss: {outputs['loss'].item():.4f}")

    # ==================== COMPLETE FLOW DIAGRAM ====================
    print_section("COMPLETE DATA FLOW SUMMARY")

    print("""
    ╔═════════════════════════════════════════════════════════════════════╗
    ║                    ZERO-BASE LLM: COMPLETE FLOW                     ║
    ╠═════════════════════════════════════════════════════════════════════╣
    ║                                                                     ║
    ║   "hello world"                                                     ║
    ║        │                                                            ║
    ║        ▼                                                            ║
    ║   ┌──────────────────────────────────────────────────────────────┐ ║
    ║   │ ZONE A: TOKENIZATION                          Shape          │ ║
    ║   │   Binary → Byte → ASCII                       (1, 11)       │ ║
    ║   └──────────────────────────────────────────────────────────────┘ ║
    ║        │                                                            ║
    ║        ▼                                                            ║
    ║   ┌──────────────────────────────────────────────────────────────┐ ║
    ║   │ ZONE B: TRANSFORMER CORE                                     │ ║
    ║   │   Embedding + 2×(Attention → FFN → LayerNorm)   (1, 11, 128) │ ║
    ║   └──────────────────────────────────────────────────────────────┘ ║
    ║        │                                                            ║
    ║        ▼                                                            ║
    ║   ┌──────────────────────────────────────────────────────────────┐ ║
    ║   │ ZONE C: WORD BUILDING                                        │ ║
    ║   │   Cluster → Project → Semantic → Context        (2, 256)    │ ║
    ║   └──────────────────────────────────────────────────────────────┘ ║
    ║        │                                                            ║
    ║        ▼                                                            ║
    ║   ┌──────────────────────────────────────────────────────────────┐ ║
    ║   │ ZONE D: SENTENCE BUILDING                                    │ ║
    ║   │   WordAttn → Grammar → Phrase → Complete        (2, 128)    │ ║
    ║   └──────────────────────────────────────────────────────────────┘ ║
    ║        │                                                            ║
    ║        ▼                                                            ║
    ║   ┌──────────────────────────────────────────────────────────────┐ ║
    ║   │ ZONE E: OUTPUT                                               │ ║
    ║   │   Project → Sample (nucleus/top-k)               next_token  │ ║
    ║   └──────────────────────────────────────────────────────────────┘ ║
    ║        │                                                            ║
    ║        ▼                                                            ║
    ║   "hello world..." (autoregressive generation)                     ║
    ║                                                                     ║
    ║   ════════════════════════════════════════════════════════════════ ║
    ║   DURING TRAINING:                                                  ║
    ║        │                                                            ║
    ║        ▼                                                            ║
    ║   ┌──────────────────────────────────────────────────────────────┐ ║
    ║   │ ZONE F: SELF-STUDY                                           │ ║
    ║   │   Forward: predict continuation                              │ ║
    ║   │   Backward: check consistency                                │ ║
    ║   │   L = L_task + 0.3×L_forward + 0.3×L_backward                │ ║
    ║   └──────────────────────────────────────────────────────────────┘ ║
    ║        │                                                            ║
    ║        ▼                                                            ║
    ║   Backpropagation → Weight Update                                  ║
    ║                                                                     ║
    ╚═════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    KEY OPERATIONS                                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   ATTENTION:     softmax(QK^T / sqrt(d_k)) @ V                     │
    │   FFN:           Linear(GELU(Linear(x)))                           │
    │   LAYER NORM:    (x - mean) / sqrt(var + eps) * gamma + beta       │
    │   CROSS ENTROPY: -sum(target * log(softmax(logits)))               │
    │   COSINE SIM:    dot(a, b) / (||a|| × ||b||)                       │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    working_diagram()