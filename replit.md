# Zero-Base LLM

A lightweight, experimental Large Language Model built from binary representation through ASCII encoding up to paragraph generation with bidirectional self-study.

## Architecture

22-layer model organized into 6 Zones:
- **Zone A (Layers 0-3):** Binary bits → ASCII characters
- **Zone B (Layers 4-8):** Transformer Core (Attention + FFN)
- **Zone C (Layers 9-12):** Character clustering → word-level representations
- **Zone D (Layers 13-18):** Sentence/paragraph coherence
- **Zone E (Layers 19-20):** Output projection and token sampling
- **Zone F (Layers 21-22):** Self-study training loop (forward/backward consistency)

## Tech Stack

- **Language:** Python 3.12
- **Framework:** PyTorch 2.11.0
- **Package manager:** pip (`.pythonlibs/`)

## Project Layout

```
zero_base_llm/
  benchmark/         # Benchmark suite (BPC, accuracy, diversity, fluency, speed)
    metrics.py       # BenchmarkSuite, BenchmarkResult, compare_results
  model/             # Neural network architecture
    layers/          # Zone implementations (zone_a.py - zone_f.py)
    modules/         # Attention, embeddings, normalization
  tokenizer/         # Binary → Byte → ASCII pipeline
  training/          # SelfStudyTrainer (cosine LR warmup, label smoothing)
  generation/        # TextGenerator, sampling strategies
  config.py          # Hyperparameter configs (small/medium/large/xl)
run.py               # CLI entry point
requirements.txt     # Dependencies
```

## Configurations

| Config | Params  | Size  | embed_dim | blocks | ff_dim |
|--------|---------|-------|-----------|--------|--------|
| small  | ~2M     | ~8MB  | 128       | 2      | 512    |
| medium | ~10.5M  | ~41MB | 256       | 6      | 1024   |
| large  | ~35M    | ~134MB| 512       | 8      | 2048   |
| xl     | ~110M   | ~420MB| 768       | 12     | 3072   |

## Optimizations Applied (v2)

- **Dropout reduced**: 0.3 → 0.1 (standard transformer value; 0.3 was over-regularizing)
- **FF dimension**: 512 → 1024 (now 4× embed_dim, GPT standard)
- **Transformer blocks**: 4 → 6 (more depth = better representations)
- **Top-k sampling**: 10 → 50 (much more diverse generation)
- **GPT-style scaled init**: residual projections scaled by 1/√(2×num_layers)
- **Label smoothing**: 0.1 (prevents over-confident predictions)
- **Cosine LR with linear warmup**: proper lr schedule for stable convergence
- **AdamW eps**: added explicit eps=1e-8 for numerical stability
- **Weight decay**: 0.01 → 0.1 (AdamW standard)
- **Self-study weights**: 0.3 → 0.2 (let task loss dominate)
- **Added XL config**: ~110M params, GPT-2 scale

## Benchmark Metrics

Same metric types used to evaluate large-scale LLMs:

| Metric | Description | Direction |
|--------|-------------|-----------|
| Bits Per Character (BPC) | Language modeling quality | ↓ lower better |
| Perplexity | exp(avg NLL), language quality | ↓ lower better |
| Top-1 Accuracy | Next token prediction accuracy | ↑ higher better |
| Top-5 Accuracy | Next token in top-5 predictions | ↑ higher better |
| Type-Token Ratio | Generation diversity | ↑ higher better |
| Trigram Repetition Rate | Degeneration detection | ↓ lower better |
| ASCII Validity Rate | Output sanity | ↑ higher better |
| Fluency Score | Word formation quality | ↑ higher better |
| Inference Speed | Tokens per second throughput | ↑ higher better |

## Usage

```bash
# Print model info with full config
python3 run.py --info
python3 run.py --info --config xl

# Run benchmark suite
python3 run.py --benchmark

# Train for 1000 steps then benchmark (shows before/after comparison)
python3 run.py --benchmark --train --steps 1000

# Single prompt generation
python3 run.py --prompt "Hello world"

# Interactive chat mode
python3 run.py --interactive

# Use a larger model
python3 run.py --config large --train --steps 2000
```

## Workflow

The "Start application" workflow runs training + benchmark to show the model learning and measure improvement.
