#!/usr/bin/env python
"""
Zero-Base LLM: Entry Point

A lightweight LLM built from binary foundation (0/1) through ASCII encoding
up to paragraph generation with bidirectional self-study.

Usage:
    python run.py                          # Interactive mode
    python run.py --prompt "Hello world"   # Single generation
    python run.py --train --steps 1000     # Self-study training
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
    generator = TextGenerator(model)
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

        # Generate response
        response = generator.chat(user_input, conversation)
        print(f"Model: {response}\n")

        # Update conversation history
        conversation.extend([user_input, response])


def generate_single(model: ZeroBaseLLM, prompt: str, max_tokens: int = 100):
    """Generate text for a single prompt."""
    generator = TextGenerator(model, GenerationConfig(max_new_tokens=max_tokens))
    output = generator.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    return output


def train_model(model: ZeroBaseLLM, steps: int = 1000, save_dir: str = None):
    """Run self-study training with diverse seed texts."""
    from zero_base_llm.training.trainer import DIVERSE_SEED_TEXTS

    trainer = SelfStudyTrainer(model, seed_texts=DIVERSE_SEED_TEXTS)
    print(f"\nStarting self-study training for {steps} steps...")
    print(f"Model has {model.count_parameters():,} parameters")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print(f"Training on {len(trainer.seed_texts)} diverse seed texts")
    print(f"Dropout rate: {model.config.attention_dropout}")

    # Show generation before training
    print("\nBefore training:")
    for prompt in ["the ", "hello ", "once upon"]:
        gen = model.generate(prompt, max_new_tokens=15, temperature=0.8)
        print(f"  '{prompt}' -> '{gen}'")
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

        # Show generation after training
        print("\nAfter training:")
        for prompt in ["the ", "hello ", "once upon", "i think", "we should"]:
            gen = model.generate(prompt, max_new_tokens=25, temperature=0.7)
            print(f"  '{prompt}' -> '{gen}'")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Base LLM - A lightweight LLM from binary foundation"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Input prompt for single generation"
    )
    parser.add_argument(
        "--max-tokens", "-m",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--train", "-t",
        action="store_true",
        help="Run self-study training"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=1000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        choices=["small", "medium", "large"],
        default="medium",
        help="Model configuration size"
    )
    parser.add_argument(
        "--checkpoint", "-k",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save model"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save training checkpoints"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive chat mode"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print model information"
    )

    args = parser.parse_args()

    # Select configuration
    if args.config == "small":
        config = ZeroBaseConfig.small()
    elif args.config == "large":
        config = ZeroBaseConfig.large()
    else:
        config = ZeroBaseConfig.medium()

    # Create model
    model = create_model(config, args.checkpoint)

    # Print model info
    if args.info:
        print(model)
        print(f"Parameters: {model.count_parameters():,}")
        print(f"Size: {model.get_model_size_mb():.2f} MB")
        return

    # Save model and exit
    if args.save and not args.train:
        model.save(args.save)
        print(f"Model saved to {args.save}")
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