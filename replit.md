# Zero-Base LLM

A fully custom language model built from the ground up — starting from binary bits (0/1) through ASCII encoding, all the way up to paragraph generation with a bidirectional self-study training loop. No borrowed pre-trained weights.

## Web Interface

The project is now a **public web app** running on port 5000 (Flask). Anyone can:
- Chat with the model
- Generate text with custom settings
- Run the benchmark suite
- Trigger self-improvement training rounds

Start command: `python3 app.py`

## Architecture

22-layer model organized into 6 Zones:

- **Zone A (Layers 0–3):** Binary bits → ASCII characters (BinaryFoundation)
- **Zone B (Layers 4–8):** Multi-head Transformer Core (GELU activation, 6 blocks)
- **Zone C (Layers 9–12):** Character clustering → word-level representations
- **Zone D (Layers 13–18):** Sentence / paragraph coherence (WordBuilder)
- **Zone E (Layers 19–20):** Output projection and nucleus sampling
- **Zone F (Layers 21–22):** Self-study training loop (forward/backward consistency)

## Tech Stack

- **Language:** Python 3.12
- **Framework:** PyTorch 2.x, Flask, Flask-CORS
- **Package manager:** pip (`.pythonlibs/`)

## Project Layout

```
app.py                    # Flask web server (port 5000)
run.py                    # CLI entry point
templates/index.html      # Full chat + generate + benchmark web UI
checkpoints/
  best_model.pt           # Saved model checkpoint
  benchmark.json          # Latest benchmark scores
zero_base_llm/
  benchmark/              # BenchmarkSuite (BPC, accuracy, diversity, fluency, speed)
  model/
    layers/               # Zone implementations (zone_a.py – zone_f.py)
    modules/              # Attention, FeedForward (GELU), normalization
  tokenizer/              # Binary → Byte → ASCII pipeline
  training/               # SelfStudyTrainer (cosine LR warmup, label smoothing)
  generation/             # TextGenerator, sampling strategies (nucleus/top-k/greedy)
  config.py               # Hyperparameter configs (small/medium/large/xl)
requirements.txt
```

## Configurations

| Config | Params  | Size   | embed_dim | blocks | ff_dim | Activation |
|--------|---------|--------|-----------|--------|--------|------------|
| small  | ~2M     | ~8MB   | 128       | 2      | 512    | GELU       |
| medium | ~10.5M  | ~41MB  | 256       | 6      | 1024   | GELU       |
| large  | ~35M    | ~134MB | 512       | 8      | 2048   | GELU       |
| xl     | ~110M   | ~420MB | 768       | 12     | 3072   | GELU       |

## Optimizations Applied (v3)

- **GELU activation** throughout (replaces ReLU — standard for GPT/Claude-style models)
- **Dropout**: 0.3 → 0.1 (standard transformer; 0.3 over-regularized)
- **FF dimension**: 512 → 1024 (4× embed_dim, GPT standard)
- **Transformer blocks**: 4 → 6 (more depth = better representations)
- **Top-k sampling**: 10 → 50 + nucleus (top-p) sampling
- **Repetition penalty**: 1.2 (avoids degenerate loops)
- **GPT-style scaled init**: residual projections scaled by 1/√(2×layers)
- **Label smoothing**: 0.1 (better calibration)
- **Cosine LR with linear warmup**: smooth learning rate schedule
- **Weight decay**: 0.1 (AdamW standard)

## Benchmark Metrics

| Metric               | Description                         | Direction    |
|----------------------|-------------------------------------|--------------|
| Bits Per Character   | Language modeling quality           | ↓ lower better |
| Perplexity           | exp(avg NLL)                        | ↓ lower better |
| Top-1 Accuracy       | Next character prediction           | ↑ higher better |
| Top-5 Accuracy       | Top-5 next character prediction     | ↑ higher better |
| Type-Token Ratio     | Generation diversity                | ↑ higher better |
| Trigram Repetition   | Degeneration detection              | ↓ lower better |
| ASCII Validity Rate  | Output sanity                       | ↑ higher better |
| Fluency Score        | Word formation quality              | ↑ higher better |
| Inference Speed      | Tokens per second                   | ↑ higher better |

## Historical Benchmark Progress

| Round | BPC  | Perplexity | Top-1 Acc | Overall Score |
|-------|------|------------|-----------|---------------|
| v1    | 7.01 | 128.6      | 0.24%     | 19.59 / 100   |
| v2    | 3.51 | 11.4       | 28.52%    | 48.77 / 100   |

## CLI Usage

```bash
python3 run.py --info                          # model info
python3 run.py --benchmark                     # benchmark suite
python3 run.py --benchmark --train --steps 1000  # train then benchmark
python3 run.py --config large --train --steps 2000
python3 run.py --prompt "Hello world"
python3 run.py --interactive
```

## Web API

```
GET  /api/info              → model info + last benchmark
POST /api/generate          → { prompt, max_tokens, temperature, strategy }
POST /api/benchmark         → run full benchmark suite
POST /api/train             → { steps } → start background training
GET  /api/train/status      → polling endpoint for training progress
```
