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
zero_base_llm/         # Core package
  model/               # Neural network architecture
    layers/            # Zone implementations (zone_a.py - zone_f.py)
    modules/           # Attention, embeddings, normalization
  tokenizer/           # Binary → Byte → ASCII pipeline
  training/            # SelfStudyTrainer
  generation/          # TextGenerator, sampling strategies
  config.py            # Hyperparameter configs (small/medium/large)
run.py                 # CLI entry point
requirements.txt       # Dependencies
```

## Usage

```bash
# Interactive chat mode (default)
python3 run.py

# Single prompt generation
python3 run.py --prompt "Hello world"

# Self-study training
python3 run.py --train --steps 1000

# Print model info
python3 run.py --info

# Select model size: small, medium (default), large
python3 run.py --config small
```

## Workflow

The "Start application" workflow runs `python3 run.py --info` as a console output to show model stats. Change the command to run in interactive, training, or generation mode as needed.
