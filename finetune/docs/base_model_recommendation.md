# Base Model Recommendation

## Decision: Qwen2.5-72B-Instruct

**Recommended base model:** `Qwen/Qwen2.5-72B-Instruct`

---

## Evaluation Summary

| Criterion             | Qwen 2.5 72B Instruct       | LLaMA 3.3 70B Instruct      |
|-----------------------|-----------------------------|-----------------------------|
| **License**           | Qwen License (commercial OK ≤100M MAU) | LLaMA 3.3 Community License (commercial OK) |
| **Coding (HumanEval)**| 86.6%                       | 88.4%                       |
| **Math (MATH)**       | 83.1%                       | 77.0%                       |
| **Math (GSM8K)**      | 95.9%                       | 95.1%                       |
| **Reasoning (BBH)**   | 86.0%                       | 81.2%                       |
| **Science (MMLU)**    | 86.1%                       | 86.0%                       |
| **Context Window**    | 128K tokens                 | 128K tokens                 |
| **Languages**         | 29 (multilingual)           | Primarily English            |
| **Flash Attn 2**      | Yes                         | Yes                         |
| **GGUF/quantization** | Excellent community support | Excellent community support |

---

## Why Qwen 2.5 72B Instruct

1. **Mathematics superiority**: +6.1 points on MATH benchmark — critical for this project's math/science/finance focus.
2. **Stronger general reasoning**: +4.8 points on BigBench Hard — benefits financial analysis and complex QA.
3. **Multilingual capability**: 29 languages out-of-the-box vs. English-primary LLaMA — useful for global financial data.
4. **Comparable coding**: Less than 2 points behind LLaMA on HumanEval; both excel on coding tasks after fine-tuning.
5. **Science/MMLU parity**: Both models score ~86% — effectively tied.
6. **Long context**: Both support 128K tokens, so no disadvantage for long document analysis.
7. **Efficiency**: Qwen 2.5 uses Grouped Query Attention (GQA) with 8 KV-heads, reducing memory footprint at inference time.
8. **License**: Both allow commercial use; Qwen's 100M MAU threshold is non-restrictive for typical deployments.

### When you might prefer LLaMA 3.3 70B
- Purely English-language applications with heavy code generation (HumanEval delta).
- Strictly open / no-restriction license requirement (LLaMA 3.3 is slightly more permissive).
- Existing Meta ecosystem tooling (llama.cpp first-class).

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install transformers>=4.45.0 accelerate bitsandbytes sentencepiece protobuf
```

### 2. Download model (Hugging Face)

```bash
# Login to Hugging Face (model is gated — requires agreement to Qwen license)
huggingface-cli login

# Download model weights (~144 GB in bf16; ~72 GB in 4-bit)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-72B-Instruct', local_dir='./models/Qwen2.5-72B-Instruct')
"
```

### 3. Verify model loads

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

inputs = tokenizer("def fibonacci(n):", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(out[0]))
```

### 4. 4-bit quantized load (for testing on smaller GPUs)

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

## Memory Requirements

| Format        | VRAM Required | Hardware            |
|---------------|---------------|---------------------|
| BF16 (full)   | ~144 GB       | 2× H100 80GB        |
| 4-bit QLoRA   | ~40-45 GB     | 1× A100 80GB ✓      |
| 4-bit QLoRA   | ~40-45 GB     | 1× H100 80GB ✓      |

**Recommendation:** Use 4-bit QLoRA for training on a single A100/H100 80GB GPU (most cost-effective cloud option).

---

## Benchmark Sources
- [Qwen2.5 Technical Report (2024)](https://qwenlm.github.io/blog/qwen2.5/)
- [LLaMA 3.3 Model Card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [Open LLM Leaderboard (Hugging Face)](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
