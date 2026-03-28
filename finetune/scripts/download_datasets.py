#!/usr/bin/env python3
"""
Dataset download, cleaning, and formatting pipeline.

Downloads open datasets for code, math, science, finance, and general
instruction-following. Outputs data in ChatML / ShareGPT format.

Usage:
    python download_datasets.py --domain all --output_dir ./data/processed
    python download_datasets.py --domain math --output_dir ./data/processed
    python download_datasets.py --domain code --limit 50000
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator

DOMAINS = ["code", "math", "science", "finance", "general"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def chatml(system: str, user: str, assistant: str) -> dict:
    """Return a record in ChatML (conversations) format."""
    return {
        "conversations": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def clean_text(text: str) -> str:
    """Basic text cleaning: strip excess whitespace."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def process_code(limit: int, output_dir: Path) -> None:
    """
    Code datasets:
      - The Stack v2 (permissive-licensed subset via BigCode smol-smoltalk proxy)
      - CodeSearchNet (Python/JS) via HuggingFace

    The Stack v2 is the primary source; CodeSearchNet fills remaining quota.
    The Stack v2 is accessed via the 'bigcode/the-stack-v2-train-smol-ids' index,
    which contains permissively-licensed file IDs. We use the pre-deduplicated
    smol (instruction-formatted) variant to avoid raw-file download complexity.
    """
    from datasets import load_dataset

    records: list[dict] = []
    system = (
        "You are an expert software engineer. Write clean, well-documented, "
        "and efficient code. Explain your reasoning when helpful."
    )

    # --- The Stack v2 (smol instruction variant) ---
    # bigcode/smol-smoltalk contains instruction-formatted coding conversations
    # drawn from The Stack v2 permissive-licensed files.
    print("  Downloading The Stack v2 (smol instruction format)...")
    STACK_TARGET = (limit * 2) // 3
    STACK_LANGUAGES = ["python", "javascript", "java", "typescript", "go", "rust"]
    try:
        ds_stack = load_dataset(
            "bigcode/smol-smoltalk",
            split="train",
            trust_remote_code=True,
        )
        for row in ds_stack:
            if len(records) >= STACK_TARGET:
                break
            convs = row.get("messages", row.get("conversations", []))
            if not convs:
                continue
            # Filter to programming-related content
            content_check = " ".join(
                m.get("content", m.get("value", "")) for m in convs
            ).lower()
            if not any(lang in content_check for lang in STACK_LANGUAGES + ["def ", "function ", "class ", "import "]):
                continue
            formatted = []
            for msg in convs:
                role = msg.get("role", msg.get("from", ""))
                content = clean_text(msg.get("content", msg.get("value", "")))
                if not content:
                    continue
                if role in ("user", "human"):
                    formatted.append({"role": "user", "content": content})
                elif role in ("assistant", "gpt"):
                    formatted.append({"role": "assistant", "content": content})
                elif role == "system":
                    formatted.append({"role": "system", "content": content})
            if len(formatted) >= 2:
                records.append({"conversations": formatted})
    except Exception as e:
        print(f"  Warning: The Stack v2 smol variant skipped ({e})")
        print("  Falling back to CodeSearchNet only.")
        STACK_TARGET = 0

    # --- The Stack v2 raw (Python subset via bigcode/the-stack-v2-dedup) ---
    # Fallback / supplement: raw Python files formatted as completion tasks
    if len(records) < STACK_TARGET:
        print("  Supplementing with The Stack v2 raw Python files...")
        try:
            ds_raw = load_dataset(
                "bigcode/the-stack-v2-dedup",
                data_files="data/python/train-*.parquet",
                split="train",
                trust_remote_code=True,
                streaming=True,
            )
            for row in ds_raw:
                if len(records) >= STACK_TARGET:
                    break
                content = clean_text(row.get("content", ""))
                if not content or len(content) < 100 or len(content) > 8000:
                    continue
                # Format as a code completion task
                lines = content.split("\n")
                # Find a reasonable split point (first function/class definition)
                split_at = 0
                for i, line in enumerate(lines):
                    if line.startswith("def ") or line.startswith("class "):
                        split_at = i
                        break
                if split_at == 0:
                    split_at = max(1, len(lines) // 4)
                prompt_lines = lines[:split_at]
                completion_lines = lines[split_at:]
                if not prompt_lines or not completion_lines:
                    continue
                user = "Complete the following Python code:\n\n```python\n" + "\n".join(prompt_lines) + "\n```"
                assistant = "```python\n" + "\n".join(prompt_lines) + "\n" + "\n".join(completion_lines) + "\n```"
                records.append(chatml(system, user, assistant))
        except Exception as e:
            print(f"  Warning: The Stack v2 raw Python skipped ({e})")

    # --- CodeSearchNet (Python) — fills remaining quota ---
    print("  Downloading CodeSearchNet (Python)...")
    csn_target = limit - len(records)
    if csn_target > 0:
        try:
            ds = load_dataset(
                "code_search_net", "python", split="train", trust_remote_code=True
            )
            for row in ds:
                if len(records) >= limit - (limit // 6):
                    break
                func_code = clean_text(row.get("func_code_string", ""))
                docstring = clean_text(row.get("func_documentation_string", ""))
                if not func_code or not docstring:
                    continue
                user = f"Write a Python function with the following description:\n\n{docstring}"
                records.append(chatml(system, user, f"```python\n{func_code}\n```"))
        except Exception as e:
            print(f"  Warning: CodeSearchNet Python skipped ({e})")

    # --- CodeSearchNet (JavaScript) ---
    print("  Downloading CodeSearchNet (JavaScript)...")
    try:
        ds_js = load_dataset(
            "code_search_net", "javascript", split="train", trust_remote_code=True
        )
        for row in ds_js:
            if len(records) >= limit:
                break
            func_code = clean_text(row.get("func_code_string", ""))
            docstring = clean_text(row.get("func_documentation_string", ""))
            if not func_code or not docstring:
                continue
            user = f"Write a JavaScript function with the following description:\n\n{docstring}"
            records.append(chatml(system, user, f"```javascript\n{func_code}\n```"))
    except Exception as e:
        print(f"  Warning: CodeSearchNet JavaScript skipped ({e})")

    save_jsonl(records[:limit], output_dir / "code.jsonl")


def process_math(limit: int, output_dir: Path) -> None:
    """
    Math datasets:
      - MATH (Hendrycks competition math)
      - GSM8K (grade school math word problems)
      - NuminaMath-CoT subset
    """
    from datasets import load_dataset

    records: list[dict] = []
    system = (
        "You are an expert mathematician. Solve problems step-by-step, "
        "showing all work clearly. Use LaTeX notation where appropriate."
    )

    # --- MATH ---
    print("  Downloading MATH dataset...")
    ds = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
    for row in ds:
        if len(records) >= limit // 3:
            break
        problem = clean_text(row.get("problem", ""))
        solution = clean_text(row.get("solution", ""))
        if not problem or not solution:
            continue
        records.append(chatml(system, problem, solution))

    # --- GSM8K ---
    print("  Downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
    for row in ds:
        if len(records) >= (2 * limit) // 3:
            break
        question = clean_text(row.get("question", ""))
        answer = clean_text(row.get("answer", ""))
        if not question or not answer:
            continue
        records.append(chatml(system, question, answer))

    # --- NuminaMath-CoT ---
    print("  Downloading NuminaMath-CoT...")
    try:
        ds = load_dataset(
            "AI-MO/NuminaMath-CoT", split="train", trust_remote_code=True
        )
        for row in ds:
            if len(records) >= limit:
                break
            problem = clean_text(row.get("problem", ""))
            solution = clean_text(row.get("solution", ""))
            if not problem or not solution:
                continue
            records.append(chatml(system, problem, solution))
    except Exception as e:
        print(f"  Warning: NuminaMath-CoT skipped ({e})")

    save_jsonl(records[:limit], output_dir / "math.jsonl")


def process_science(limit: int, output_dir: Path) -> None:
    """
    Science datasets:
      - SciQ (science QA)
      - ARC (AI2 Reasoning Challenge)
    """
    from datasets import load_dataset

    records: list[dict] = []
    system = (
        "You are a knowledgeable scientist and educator. Provide accurate, "
        "detailed explanations grounded in scientific evidence."
    )

    # --- SciQ ---
    print("  Downloading SciQ...")
    ds = load_dataset("allenai/sciq", split="train", trust_remote_code=True)
    for row in ds:
        if len(records) >= limit // 2:
            break
        question = clean_text(row.get("question", ""))
        answer = clean_text(row.get("correct_answer", ""))
        support = clean_text(row.get("support", ""))
        if not question or not answer:
            continue
        full_answer = f"{answer}\n\n{support}" if support else answer
        records.append(chatml(system, question, full_answer))

    # --- ARC Challenge ---
    print("  Downloading ARC Challenge...")
    ds = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split="train", trust_remote_code=True
    )
    for row in ds:
        if len(records) >= limit:
            break
        question = clean_text(row.get("question", ""))
        choices = row.get("choices", {})
        answer_key = row.get("answerKey", "")
        if not question or not choices or not answer_key:
            continue
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        options_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        try:
            answer_idx = labels.index(answer_key)
            answer_text = texts[answer_idx]
        except (ValueError, IndexError):
            continue
        user = f"{question}\n\n{options_str}"
        assistant = f"The correct answer is {answer_key}. {answer_text}"
        records.append(chatml(system, user, assistant))

    save_jsonl(records[:limit], output_dir / "science.jsonl")


def process_finance(limit: int, output_dir: Path) -> None:
    """
    Finance datasets:
      - FinGPT / financial-phrasebank (sentiment)
      - financial_qa datasets
      - Alpaca-finance subset
    """
    from datasets import load_dataset

    records: list[dict] = []
    system = (
        "You are an expert financial analyst with deep knowledge of markets, "
        "accounting, economics, and investment strategy. Provide clear, accurate "
        "financial analysis. Always note when assumptions are made."
    )

    # --- Financial PhraseBank (sentiment analysis) ---
    print("  Downloading Financial PhraseBank...")
    try:
        ds = load_dataset(
            "financial_phrasebank", "sentences_allagree",
            split="train", trust_remote_code=True
        )
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        for row in ds:
            if len(records) >= limit // 3:
                break
            sentence = clean_text(row.get("sentence", ""))
            label = label_map.get(row.get("label", -1))
            if not sentence or not label:
                continue
            user = f"What is the financial sentiment of the following text?\n\n\"{sentence}\""
            assistant = (
                f"The financial sentiment is **{label}**.\n\n"
                f"Analysis: The text {'indicates a positive outlook' if label == 'positive' else 'suggests a negative outlook' if label == 'negative' else 'presents a neutral or mixed perspective'} "
                f"for the company or market segment described."
            )
            records.append(chatml(system, user, assistant))
    except Exception as e:
        print(f"  Warning: Financial PhraseBank skipped ({e})")

    # --- FinQA ---
    print("  Downloading FinQA...")
    try:
        ds = load_dataset("dreamerdeo/finqa", split="train", trust_remote_code=True)
        for row in ds:
            if len(records) >= (2 * limit) // 3:
                break
            question = clean_text(row.get("question", ""))
            answer = clean_text(str(row.get("answer", "")))
            context = clean_text(row.get("context", ""))
            if not question or not answer:
                continue
            user_text = f"{context}\n\nQuestion: {question}" if context else question
            records.append(chatml(system, user_text, answer))
    except Exception as e:
        print(f"  Warning: FinQA skipped ({e})")

    # --- Finance Alpaca ---
    print("  Downloading Finance Alpaca...")
    try:
        ds = load_dataset(
            "gbharti/finance-alpaca", split="train", trust_remote_code=True
        )
        for row in ds:
            if len(records) >= limit:
                break
            instruction = clean_text(row.get("instruction", ""))
            inp = clean_text(row.get("input", ""))
            output = clean_text(row.get("output", ""))
            if not instruction or not output:
                continue
            user = f"{instruction}\n\n{inp}" if inp else instruction
            records.append(chatml(system, user, output))
    except Exception as e:
        print(f"  Warning: Finance Alpaca skipped ({e})")

    save_jsonl(records[:limit], output_dir / "finance.jsonl")


def process_general(limit: int, output_dir: Path) -> None:
    """
    General instruction-following datasets:
      - OpenHermes 2.5
      - ShareGPT (cleaned subset via anon8231489123/ShareGPT_Vicuna_unfiltered)
    """
    from datasets import load_dataset

    records: list[dict] = []

    # --- OpenHermes 2.5 ---
    print("  Downloading OpenHermes-2.5...")
    try:
        ds = load_dataset(
            "teknium/OpenHermes-2.5", split="train", trust_remote_code=True
        )
        for row in ds:
            if len(records) >= limit // 2:
                break
            convs = row.get("conversations", [])
            if not convs:
                continue
            formatted = []
            for msg in convs:
                role = msg.get("from", msg.get("role", ""))
                content = clean_text(msg.get("value", msg.get("content", "")))
                if role in ("human", "user"):
                    formatted.append({"role": "user", "content": content})
                elif role in ("gpt", "assistant"):
                    formatted.append({"role": "assistant", "content": content})
                elif role == "system":
                    formatted.append({"role": "system", "content": content})
            if len(formatted) >= 2:
                records.append({"conversations": formatted})
    except Exception as e:
        print(f"  Warning: OpenHermes-2.5 skipped ({e})")

    # --- ShareGPT cleaned ---
    print("  Downloading ShareGPT (cleaned)...")
    try:
        ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            split="train",
            trust_remote_code=True,
        )
        for row in ds:
            if len(records) >= limit:
                break
            convs = row.get("conversations", [])
            if not convs:
                continue
            formatted = []
            for msg in convs:
                role = msg.get("from", "")
                content = clean_text(msg.get("value", ""))
                if not content:
                    continue
                if role in ("human", "user"):
                    formatted.append({"role": "user", "content": content})
                elif role in ("gpt", "assistant"):
                    formatted.append({"role": "assistant", "content": content})
                elif role == "system":
                    formatted.append({"role": "system", "content": content})
            if len(formatted) >= 2:
                records.append({"conversations": formatted})
    except Exception as e:
        print(f"  Warning: ShareGPT skipped ({e})")

    save_jsonl(records[:limit], output_dir / "general.jsonl")


def merge_all(output_dir: Path, limit_per_domain: int) -> None:
    """Merge all domain JSONL files into a single shuffle-ready combined file."""
    import random

    all_records: list[dict] = []
    for domain in DOMAINS:
        path = output_dir / f"{domain}.jsonl"
        if not path.exists():
            print(f"  Skipping {domain} (file not found)")
            continue
        with open(path, "r", encoding="utf-8") as f:
            domain_records = [json.loads(line) for line in f if line.strip()]
        print(f"  {domain}: {len(domain_records):,} records")
        all_records.extend(domain_records)

    random.shuffle(all_records)
    save_jsonl(all_records, output_dir / "combined.jsonl")
    print(f"\nCombined dataset: {len(all_records):,} total records")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset download and formatting pipeline")
    parser.add_argument(
        "--domain",
        choices=DOMAINS + ["all"],
        default="all",
        help="Which domain to download (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50_000,
        help="Max records per domain (default: 50000)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="After downloading, merge all domains into combined.jsonl",
    )
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: pip install datasets")
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    domains_to_run = DOMAINS if args.domain == "all" else [args.domain]
    processors = {
        "code": process_code,
        "math": process_math,
        "science": process_science,
        "finance": process_finance,
        "general": process_general,
    }

    for domain in domains_to_run:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain.upper()}")
        print(f"{'='*60}")
        processors[domain](args.limit, output_dir)

    if args.merge or args.domain == "all":
        print(f"\n{'='*60}")
        print("Merging all domains...")
        print(f"{'='*60}")
        merge_all(output_dir, args.limit)

    print("\nDataset pipeline complete.")


if __name__ == "__main__":
    main()
