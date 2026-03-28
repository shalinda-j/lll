#!/usr/bin/env python3
"""
Merge trained LoRA adapter into base model weights and export.

Exports in two formats:
  1. Hugging Face format (full bf16 model, ready for inference / further fine-tuning)
  2. GGUF format (quantized, for CPU / low-power inference with llama.cpp)

Usage:
    python merge_and_export.py \
        --adapter_dir ./outputs/code/final_adapter \
        --base_model Qwen/Qwen2.5-72B-Instruct \
        --output_dir ./exports/code-merged \
        --gguf_quant Q4_K_M

    # HF export only (skip GGUF conversion)
    python merge_and_export.py \
        --adapter_dir ./outputs/math/final_adapter \
        --base_model Qwen/Qwen2.5-72B-Instruct \
        --output_dir ./exports/math-merged \
        --skip_gguf
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


SUPPORTED_QUANTS = [
    "Q4_K_M",  # Recommended: 4-bit, medium quality (~40GB for 72B)
    "Q5_K_M",  # 5-bit, better quality (~50GB for 72B)
    "Q8_0",    # 8-bit, near-lossless (~75GB for 72B)
    "Q2_K",    # 2-bit, smallest but lower quality
    "F16",     # Full 16-bit (no quantization, largest)
]


def check_dependencies():
    missing = []
    for pkg in ["torch", "transformers", "peft"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print("Run: pip install -r ../requirements_finetune.txt")
        sys.exit(1)


def merge_lora_into_base(
    adapter_dir: str,
    base_model: str,
    output_dir: str,
) -> None:
    """Load base model + LoRA adapter, merge weights, save full model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from adapter: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    print(f"Loading base model: {base_model}")
    print("  (This requires ~144GB VRAM/RAM for bf16 — ensure sufficient memory)")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model → {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)

    print(f"HF model saved: {output_dir}")
    print(f"  Files: {list(output_path.glob('*.safetensors'))[:3]} ...")


def convert_to_gguf(
    hf_model_dir: str,
    output_dir: str,
    quant_type: str = "Q4_K_M",
) -> str:
    """
    Convert Hugging Face model to GGUF using llama.cpp's convert script.

    Requires llama.cpp to be cloned and built at ./llama.cpp
    or installed as a Python package (llama-cpp-python).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gguf_base_path = output_path / "model_f16.gguf"
    gguf_quant_path = output_path / f"model_{quant_type}.gguf"

    # Look for llama.cpp convert script
    convert_scripts = [
        "./llama.cpp/convert_hf_to_gguf.py",
        "./llama.cpp/convert-hf-to-gguf.py",
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
    ]
    convert_script = None
    for path in convert_scripts:
        if os.path.exists(path):
            convert_script = path
            break

    if convert_script is None:
        print("\nWARNING: llama.cpp convert script not found.")
        print("To enable GGUF conversion, clone and build llama.cpp:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && pip install -r requirements.txt")
        print(f"\nThen re-run with --hf_model_dir {hf_model_dir}")
        return ""

    # Step 1: Convert HF → GGUF (F16)
    print(f"\nStep 1: Converting HF model → GGUF (F16)...")
    cmd_convert = [
        sys.executable,
        convert_script,
        hf_model_dir,
        "--outfile", str(gguf_base_path),
        "--outtype", "f16",
    ]
    print(f"  Running: {' '.join(cmd_convert)}")
    result = subprocess.run(cmd_convert, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Conversion failed (exit code {result.returncode})")
        sys.exit(1)

    # Step 2: Quantize F16 GGUF → target quantization
    quantize_binaries = [
        "./llama.cpp/llama-quantize",
        "./llama.cpp/quantize",
        os.path.expanduser("~/llama.cpp/llama-quantize"),
    ]
    quantize_bin = None
    for path in quantize_binaries:
        if os.path.exists(path):
            quantize_bin = path
            break

    if quantize_bin is None:
        print(f"\nWARNING: llama-quantize binary not found.")
        print("Build llama.cpp first: cd llama.cpp && make")
        print(f"F16 GGUF saved at: {gguf_base_path}")
        return str(gguf_base_path)

    print(f"\nStep 2: Quantizing GGUF → {quant_type}...")
    cmd_quant = [
        quantize_bin,
        str(gguf_base_path),
        str(gguf_quant_path),
        quant_type,
    ]
    print(f"  Running: {' '.join(cmd_quant)}")
    result = subprocess.run(cmd_quant, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Quantization failed (exit code {result.returncode})")
        sys.exit(1)

    print(f"\nGGUF model saved: {gguf_quant_path}")
    print(f"  Size: {gguf_quant_path.stat().st_size / 1e9:.1f} GB")

    # Optionally remove the large F16 intermediary
    if quant_type != "F16" and gguf_base_path.exists():
        print(f"Removing F16 intermediary: {gguf_base_path}")
        gguf_base_path.unlink()

    return str(gguf_quant_path)


def print_usage_guide(hf_dir: str, gguf_path: str) -> None:
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE — Usage Guide")
    print("=" * 60)

    if hf_dir:
        print(f"\n1. Hugging Face format: {hf_dir}")
        print("   Load with:")
        print("   ```python")
        print("   from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f'   model = AutoModelForCausalLM.from_pretrained("{hf_dir}")')
        print(f'   tokenizer = AutoTokenizer.from_pretrained("{hf_dir}")')
        print("   ```")

    if gguf_path and os.path.exists(gguf_path):
        print(f"\n2. GGUF format: {gguf_path}")
        print("   Load with llama.cpp:")
        print("   ```bash")
        print(f"   ./llama.cpp/llama-cli -m {gguf_path} -p 'Hello' -n 200")
        print("   ```")
        print("   Or with Python (llama-cpp-python):")
        print("   ```python")
        print("   from llama_cpp import Llama")
        print(f'   llm = Llama(model_path="{gguf_path}", n_ctx=4096)')
        print('   out = llm("def fibonacci(n):", max_tokens=200)')
        print("   ```")

    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and export"
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to saved LoRA adapter (output of train.py)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Base model name or path (default: Qwen/Qwen2.5-72B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--gguf_quant",
        type=str,
        default="Q4_K_M",
        choices=SUPPORTED_QUANTS,
        help=f"GGUF quantization type (default: Q4_K_M). Options: {SUPPORTED_QUANTS}",
    )
    parser.add_argument(
        "--skip_gguf",
        action="store_true",
        help="Skip GGUF conversion (HF format only)",
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Skip merge step (use pre-merged HF model for GGUF conversion only)",
    )
    args = parser.parse_args()

    check_dependencies()

    hf_output = args.output_dir
    gguf_path = ""

    if not args.skip_merge:
        print("\n" + "=" * 60)
        print("Step 1/2: Merging LoRA adapter into base model")
        print("=" * 60)
        merge_lora_into_base(
            adapter_dir=args.adapter_dir,
            base_model=args.base_model,
            output_dir=hf_output,
        )
    else:
        print(f"Skipping merge; using existing HF model at: {hf_output}")

    if not args.skip_gguf:
        print("\n" + "=" * 60)
        print("Step 2/2: Converting to GGUF format")
        print("=" * 60)
        gguf_dir = os.path.join(hf_output, "gguf")
        gguf_path = convert_to_gguf(
            hf_model_dir=hf_output,
            output_dir=gguf_dir,
            quant_type=args.gguf_quant,
        )

    print_usage_guide(hf_output, gguf_path)


if __name__ == "__main__":
    main()
