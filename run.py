#!/usr/bin/env python
"""
Zero-Base LLM: Entry Point

A lightweight LLM built from binary foundation (0/1) through ASCII encoding
up to paragraph generation with bidirectional self-study.

Usage:
    python run.py                          # Interactive mode
    python run.py --info                   # Print model info
    python run.py --prompt "Hello world"   # Single generation
    python run.py --train --steps 1000     # Self-study training
    python run.py --benchmark              # Run full benchmark suite
    python run.py --benchmark --train      # Train then benchmark (before/after)
    python run.py --save model.pt          # Save model
"""

import argparse
import sys
from pathlib import Path

import torch

from zero_base_llm import ZeroBaseConfig, ZeroBaseLLM
from zero_base_llm.training import SelfStudyTrainer
from zero_base_llm.generation import TextGenerator, GenerationConfig


def create_model(config: ZeroBaseConfig = None, checkpoint: str = None) -> ZeroBaseLLM:
    """Create or load a model."""
    if checkpoint and Path(checkpoint).exists():
        print(f"Loading model from {checkpoint}")
        return ZeroBaseLLM.load(checkpoint)

    return ZeroBaseLLM(config or ZeroBaseConfig())


def interactive_mode(model: ZeroBaseLLM):
    """Run in interactive chat mode."""
    gen_cfg = GenerationConfig(
        max_new_tokens=150,
        temperature=0.8,
        top_k=50,
        top_p=0.92,
        strategy="nucleus",
        repetition_penalty=1.15,
    )
    generator = TextGenerator(model, gen_cfg)
    print("\n" + "=" * 50)
    print("Zero-Base LLM - Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = generator.chat(user_input, conversation)
        clean = "".join(c if c.isprintable() else "" for c in response)
        print(f"Model: {clean}\n")

        conversation.extend([user_input, response])


def generate_single(model: ZeroBaseLLM, prompt: str, max_tokens: int = 100):
    """Generate text for a single prompt."""
    gen_cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=50,
        top_p=0.92,
        strategy="nucleus",
        repetition_penalty=1.15,
    )
    generator = TextGenerator(model, gen_cfg)
    output = generator.generate(prompt)
    clean = "".join(c if c.isprintable() else "" for c in output)
    print(f"Prompt: {prompt}")
    print(f"Output: {clean}")
    return output


def train_model(model: ZeroBaseLLM, steps: int = 1000, save_dir: str = None):
    """Run self-study training with diverse seed texts."""
    from zero_base_llm.training.trainer import DIVERSE_SEED_TEXTS

    trainer = SelfStudyTrainer(model, seed_texts=DIVERSE_SEED_TEXTS)
    print(f"\nStarting self-study training for {steps} steps...")
    print(f"Model       : {model.count_parameters():,} parameters")
    print(f"Size        : {model.get_model_size_mb():.2f} MB")
    print(f"Seed texts  : {len(trainer.seed_texts)} sentences")
    print(f"Dropout     : {model.config.attention_dropout}")
    print(f"Warmup steps: {trainer.warmup_steps}")
    print(f"LR          : {model.config.learning_rate}")

    print("\nBefore training:")
    for prompt in ["the ", "hello ", "once upon"]:
        gen = model.generate(prompt, max_new_tokens=15, temperature=0.8)
        clean = "".join(c if c.isprintable() else "?" for c in gen[:40])
        print(f"  '{prompt}' -> '{clean}'")
    print()

    history = trainer.train(
        num_steps=steps,
        log_interval=max(100, steps // 20),
        generate_interval=max(500, steps // 10),
        save_interval=max(1000, steps // 5),
        save_dir=save_dir
    )

    if history["loss"]:
        print(f"\nTraining complete!")
        print(f"Final loss: {history['loss'][-1]:.4f}")

        print("\nAfter training:")
        for prompt in ["the ", "hello ", "once upon", "i think", "we should"]:
            gen = model.generate(prompt, max_new_tokens=25, temperature=0.8)
            clean = "".join(c if c.isprintable() else "?" for c in gen[:60])
            print(f"  '{prompt}' -> '{clean}'")

    return model


def run_benchmark(model: ZeroBaseLLM, before_result=None):
    """Run the benchmark suite and optionally compare with a before result."""
    from zero_base_llm.benchmark import BenchmarkSuite
    from zero_base_llm.benchmark.metrics import compare_results

    suite = BenchmarkSuite(model)
    result = suite.run(verbose=True)
    print(result)

    if before_result is not None:
        print(compare_results(before_result, result))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Base LLM - A lightweight LLM from binary foundation"
    )
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Input prompt for single generation")
    parser.add_argument("--max-tokens", "-m", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--train", "-t", action="store_true",
                        help="Run self-study training")
    parser.add_argument("--steps", "-s", type=int, default=1000,
                        help="Number of training steps")
    parser.add_argument("--config", "-c", type=str,
                        choices=["small", "medium", "large", "xl"],
                        default="medium",
                        help="Model configuration size")
    parser.add_argument("--checkpoint", "-k", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save model")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save training checkpoints")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive chat mode")
    parser.add_argument("--info", action="store_true",
                        help="Print model information")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Run full benchmark suite")

    args = parser.parse_args()

    # Select configuration
    config_map = {
        "small":  ZeroBaseConfig.small(),
        "medium": ZeroBaseConfig.medium(),
        "large":  ZeroBaseConfig.large(),
        "xl":     ZeroBaseConfig.xl(),
    }
    config = config_map[args.config]

    # Create model
    model = create_model(config, args.checkpoint)

    # Print model info
    if args.info:
        print(model)
        print(f"Parameters : {model.count_parameters():,}")
        print(f"Size       : {model.get_model_size_mb():.2f} MB")
        print(f"Config     : {args.config}")
        cfg = model.config
        print(f"  embed_dim       : {cfg.embed_dim}")
        print(f"  num_heads       : {cfg.num_heads}")
        print(f"  ff_dim          : {cfg.ff_dim}")
        print(f"  num_blocks      : {cfg.num_transformer_blocks}")
        print(f"  dropout         : {cfg.attention_dropout}")
        print(f"  top_k           : {cfg.top_k}")
        print(f"  top_p           : {cfg.top_p}")
        print(f"  warmup_steps    : {cfg.warmup_steps}")
        print(f"  label_smoothing : {cfg.label_smoothing}")
        return

    # Save model and exit
    if args.save and not args.train:
        model.save(args.save)
        print(f"Model saved to {args.save}")
        return

    # Benchmark-only mode
    if args.benchmark and not args.train:
        run_benchmark(model)
        return

    # Benchmark before training, train, then benchmark after
    if args.benchmark and args.train:
        print("Running pre-training benchmark...")
        before = run_benchmark(model)

        model = train_model(model, args.steps, args.save_dir)
        if args.save:
            model.save(args.save)
            print(f"Model saved to {args.save}")

        print("\nRunning post-training benchmark...")
        run_benchmark(model, before_result=before)
        return

    # Training mode
    if args.train:
        model = train_model(model, args.steps, args.save_dir)
        if args.save:
            model.save(args.save)
            print(f"Final model saved to {args.save}")
        return

    # Single generation mode
    if args.prompt:
        generate_single(model, args.prompt, args.max_tokens)
        return

    # Interactive mode (default)
    interactive_mode(model)


if __name__ == "__main__":
    main()
