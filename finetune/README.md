# LLM Fine-Tuning Pipeline

QLoRA fine-tuning infrastructure for **Qwen2.5-72B-Instruct** across five specialized domains: coding, mathematics, science, finance, and general reasoning.

---

## Quick Start

```bash
# 1. (On cloud GPU) Install dependencies
pip install -r requirements_finetune.txt
pip install flash-attn --no-build-isolation

# 2. Download datasets for your target domain
python scripts/download_datasets.py --domain math --output_dir data/processed

# 3. Run fine-tuning
python scripts/train.py --config configs/math.yaml

# 4. Merge adapter into base model and export
python scripts/merge_and_export.py \
    --adapter_dir outputs/math/final_adapter \
    --base_model Qwen/Qwen2.5-72B-Instruct \
    --output_dir exports/math-merged \
    --gguf_quant Q4_K_M
```

---

## Directory Structure

```
finetune/
├── README.md                        # This file
├── requirements_finetune.txt        # Python dependencies
├── configs/
│   ├── base.yaml                    # Default hyperparameters (inherited)
│   ├── code.yaml                    # Code fine-tuning config
│   ├── math.yaml                    # Math fine-tuning config
│   ├── science.yaml                 # Science fine-tuning config
│   ├── finance.yaml                 # Finance fine-tuning config
│   ├── general.yaml                 # General instruction config
│   └── combined.yaml                # All domains mixed
├── scripts/
│   ├── download_datasets.py         # Download + format training data
│   ├── train.py                     # QLoRA fine-tuning script
│   └── merge_and_export.py          # Merge LoRA + export HF/GGUF
├── docs/
│   ├── base_model_recommendation.md # Model selection analysis
│   └── cloud_gpu_setup_guide.md     # Full cloud GPU tutorial
└── data/                            # Created by download_datasets.py
    └── processed/
        ├── code.jsonl
        ├── math.jsonl
        ├── science.jsonl
        ├── finance.jsonl
        ├── general.jsonl
        └── combined.jsonl           # Merged (with --merge flag)
```

---

## Datasets Used

| Domain | Datasets | Format |
|--------|----------|--------|
| **Code** | CodeSearchNet (Python, JS) | Function + docstring pairs |
| **Math** | MATH (Hendrycks), GSM8K, NuminaMath-CoT | Problem + step-by-step solution |
| **Science** | SciQ, ARC-Challenge | Question + evidence-backed answer |
| **Finance** | Financial PhraseBank, FinQA, Finance Alpaca | Sentiment, numerical QA, analysis |
| **General** | OpenHermes-2.5, ShareGPT | Multi-turn chat conversations |

All data is formatted in **ChatML** format for compatibility with Qwen2.5's chat template.

---

## Training Configuration

All configs share these defaults (see `configs/base.yaml`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen2.5-72B-Instruct | |
| Quantization | 4-bit NF4 (QLoRA) | ~40-45GB VRAM |
| LoRA rank | 64 | `r=64, alpha=128` |
| LoRA targets | q/k/v/o/gate/up/down proj | All attention + FFN layers |
| Max sequence | 4096 tokens | Code uses 8192 |
| Effective batch | 16 | 2 × 8 grad accum |
| Learning rate | 1.5-2.0e-4 | Cosine schedule |
| Attention | Flash Attention 2 | Required: A100/H100 |
| Optimizer | paged_adamw_32bit | Memory-efficient |

---

## Output Formats

After training:

1. **LoRA adapter** (`outputs/<domain>/final_adapter/`) — Small (~1-2GB). Contains only the trained delta weights. Use with the original base model.

2. **Merged HF model** (`exports/<domain>-merged/`) — Full merged model in safetensors format. Use directly with `transformers`.

3. **GGUF model** (`exports/<domain>-merged/gguf/model_Q4_K_M.gguf`) — Quantized for CPU/low-power inference with `llama.cpp` or `llama-cpp-python`.

---

## Documentation

- **Model selection rationale:** `docs/base_model_recommendation.md`
- **Cloud GPU tutorial (RunPod/Lambda Labs):** `docs/cloud_gpu_setup_guide.md`

---

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Training (QLoRA 4-bit) | A100 40GB | A100 80GB or H100 80GB |
| Inference (merged bf16) | 2× A100 80GB | 2× H100 80GB |
| Inference (GGUF Q4_K_M) | 48GB RAM (CPU) | Apple M2 Ultra / Mac Studio |
