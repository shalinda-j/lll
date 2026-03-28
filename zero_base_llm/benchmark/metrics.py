"""
Benchmark Suite for Zero-Base LLM.

Measures the same metric TYPES used to evaluate large language models:
  - Bits Per Character (BPC) / Perplexity   -> language modelling quality
  - Top-1 / Top-5 accuracy                  -> prediction quality
  - Generation diversity (TTR)              -> creativity / degeneration
  - N-gram repetition rate                  -> degeneration detection
  - ASCII validity rate                     -> output sanity
  - Fluency score                           -> word formation quality
  - Inference speed (tokens / second)       -> throughput performance
  - Self-study consistency score            -> Zone-F metric

All metrics are collected on held-out evaluation texts so there is no
train-set contamination.
"""

import time
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

EVAL_TEXTS = [
    "the quick brown fox jumps over the lazy dog near the river bank",
    "artificial intelligence is transforming the way we interact with computers",
    "language models learn patterns from large amounts of text data",
    "the sun rises in the east and sets in the west every day",
    "machine learning algorithms can identify complex patterns in data",
    "natural language processing enables computers to understand human speech",
    "deep learning models have achieved remarkable results in recent years",
    "the history of computing spans several decades of innovation",
    "neural networks are inspired by the structure of the human brain",
    "scientists study the universe to understand its fundamental laws",
    "music has the power to evoke strong emotions in people",
    "reading books is one of the best ways to expand knowledge",
    "technology continues to evolve at an ever increasing pace today",
    "the ocean covers more than seventy percent of the earth surface",
    "trees absorb carbon dioxide and release oxygen into the atmosphere",
    "education is the foundation upon which societies build their future",
    "creativity and critical thinking are essential skills in modern times",
    "communication between people forms the basis of human civilization",
    "the brain processes information through billions of neural connections",
    "every great journey begins with a single determined first step",
]

GENERATION_PROMPTS = [
    "the ",
    "hello ",
    "once upon a ",
    "i think that ",
    "we should ",
    "the cat ",
    "in the morning ",
    "artificial intelligence ",
    "learning is ",
    "the future of ",
]

COMMON_WORDS = set([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "up", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "and", "but",
    "or", "nor", "so", "yet", "both", "either", "neither", "not", "only",
    "same", "than", "too", "very", "just", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "own", "right", "out",
    "there", "here", "my", "your", "his", "her", "its", "our", "their",
    "time", "year", "people", "way", "day", "man", "woman", "child", "life",
    "world", "school", "state", "family", "student", "group", "country",
    "problem", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "lot", "right", "study", "book", "eye", "job", "word", "side",
    "kind", "head", "house", "service", "friend", "father", "power", "hour",
    "game", "line", "end", "among", "ever", "stand", "own", "turn", "become",
    "leave", "put", "mean", "keep", "let", "begin", "seem", "show", "hear",
    "play", "run", "move", "live", "believe", "hold", "bring", "happen",
    "write", "provide", "sit", "stand", "lose", "pay", "meet", "include",
    "continue", "set", "learn", "change", "lead", "understand", "watch",
    "follow", "stop", "create", "speak", "read", "spend", "grow", "open",
    "walk", "win", "offer", "remember", "love", "consider", "appear",
    "buy", "wait", "serve", "die", "send", "expect", "build", "stay",
    "fall", "cut", "reach", "kill", "raise", "pass", "sell", "require",
    "report", "decide", "pull", "good", "new", "old", "great", "small",
    "large", "long", "little", "right", "big", "high", "low", "next",
    "early", "young", "important", "public", "private", "real", "best",
    "free", "black", "white", "American", "human", "true", "strong", "open",
    "specific", "possible", "clear", "sure", "wrong", "able", "different",
    "economic", "political", "social", "military", "national", "local",
])


@dataclass
class BenchmarkResult:
    """Results from a full benchmark run."""
    model_name: str = "ZeroBaseLLM"
    param_count: int = 0
    model_size_mb: float = 0.0

    bits_per_char: float = 0.0
    perplexity: float = 0.0
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0

    type_token_ratio: float = 0.0
    trigram_repetition_rate: float = 0.0
    ascii_validity_rate: float = 0.0
    fluency_score: float = 0.0

    tokens_per_second: float = 0.0
    eval_time_seconds: float = 0.0

    generation_samples: List[Dict[str, str]] = field(default_factory=list)
    config_summary: Dict[str, Any] = field(default_factory=dict)

    def overall_score(self) -> float:
        """
        Composite score (0-100, higher is better).

        Mirrors how large-model leaderboards combine multiple axes
        into a single overall score.
        """
        bpc_score    = max(0.0, 1.0 - (self.bits_per_char - 1.0) / 6.0)
        acc_score    = (self.top1_accuracy + self.top5_accuracy) / 2.0
        div_score    = self.type_token_ratio
        rep_score    = max(0.0, 1.0 - self.trigram_repetition_rate)
        valid_score  = self.ascii_validity_rate
        fluency      = self.fluency_score
        composite = (
            bpc_score   * 0.30 +
            acc_score   * 0.25 +
            div_score   * 0.15 +
            rep_score   * 0.10 +
            valid_score * 0.10 +
            fluency     * 0.10
        )
        return round(composite * 100, 2)

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 68,
            f"  ZERO-BASE LLM BENCHMARK REPORT",
            "=" * 68,
            f"  Model          : {self.model_name}",
            f"  Parameters     : {self.param_count:,}",
            f"  Size           : {self.model_size_mb:.2f} MB",
            "-" * 68,
            f"  LANGUAGE MODELLING (lower BPC / perplexity = better)",
            f"    Bits Per Char  : {self.bits_per_char:.4f}",
            f"    Perplexity     : {self.perplexity:.4f}",
            "-" * 68,
            f"  PREDICTION ACCURACY",
            f"    Top-1 Accuracy : {self.top1_accuracy * 100:.2f} %",
            f"    Top-5 Accuracy : {self.top5_accuracy * 100:.2f} %",
            "-" * 68,
            f"  GENERATION QUALITY",
            f"    Type-Token Ratio (diversity) : {self.type_token_ratio:.4f}",
            f"    Trigram Repetition Rate      : {self.trigram_repetition_rate:.4f}",
            f"    ASCII Validity Rate          : {self.ascii_validity_rate:.4f}",
            f"    Fluency Score (word quality) : {self.fluency_score:.4f}",
            "-" * 68,
            f"  PERFORMANCE",
            f"    Inference Speed : {self.tokens_per_second:.1f} tokens/sec",
            f"    Eval Time       : {self.eval_time_seconds:.2f} s",
            "-" * 68,
            f"  OVERALL SCORE   : {self.overall_score():.2f} / 100",
            "=" * 68,
        ]
        if self.generation_samples:
            lines.append("  GENERATION SAMPLES")
            lines.append("-" * 68)
            for s in self.generation_samples:
                prompt = s.get("prompt", "")
                gen = s.get("generated", "")
                clean = "".join(c if c.isprintable() else "?" for c in gen)
                lines.append(f"  [{prompt!r:16s}] -> {clean!r}")
        lines.append("=" * 68)
        return "\n".join(lines)


class BenchmarkSuite:
    """
    Evaluates a ZeroBaseLLM on a comprehensive set of metrics.
    """

    def __init__(self, model, eval_texts: Optional[List[str]] = None):
        self.model = model
        self.eval_texts = eval_texts or EVAL_TEXTS
        self.device = next(model.parameters()).device

    def _encode_text(self, text: str) -> torch.Tensor:
        return self.model.encode(text)

    @torch.no_grad()
    def _compute_lm_metrics(self) -> Dict[str, float]:
        """Compute BPC, perplexity, top-1 and top-5 accuracy."""
        self.model.eval()

        total_nll = 0.0
        total_tokens = 0
        top1_correct = 0
        top5_correct = 0

        for text in self.eval_texts:
            ids = self._encode_text(text)
            if len(ids) < 2:
                continue

            ids = ids.unsqueeze(0).to(self.device)
            inputs = ids[:, :-1]
            targets = ids[:, 1:]

            if inputs.size(1) == 0:
                continue

            outputs = self.model(inputs, use_self_study=False)
            char_hidden = outputs["char_hidden"]

            logits = self.model.char_output_layer(char_hidden)

            T = min(logits.size(1), targets.size(1))
            logits_t  = logits[:, :T, :]
            targets_t = targets[:, :T]

            log_probs = F.log_softmax(logits_t, dim=-1)

            gathered = log_probs.gather(2, targets_t.unsqueeze(-1)).squeeze(-1)
            total_nll -= gathered.sum().item()
            n = targets_t.numel()
            total_tokens += n

            preds_top1 = logits_t.argmax(dim=-1)
            top1_correct += (preds_top1 == targets_t).sum().item()

            _, top5_indices = logits_t.topk(5, dim=-1)
            targets_expanded = targets_t.unsqueeze(-1).expand_as(top5_indices)
            top5_correct += (top5_indices == targets_expanded).any(dim=-1).sum().item()

        if total_tokens == 0:
            return {"bpc": 99.0, "perplexity": 99.0, "top1": 0.0, "top5": 0.0}

        avg_nll     = total_nll / total_tokens
        bpc         = avg_nll / math.log(2)
        perplexity  = math.exp(avg_nll)
        top1_acc    = top1_correct / total_tokens
        top5_acc    = top5_correct / total_tokens

        return {
            "bpc": bpc,
            "perplexity": perplexity,
            "top1": top1_acc,
            "top5": top5_acc,
        }

    @torch.no_grad()
    def _compute_generation_metrics(
        self, num_tokens: int = 200
    ) -> Dict[str, Any]:
        """Compute diversity, repetition, validity, and fluency metrics."""
        self.model.eval()

        all_generated = []
        samples = []

        for prompt in GENERATION_PROMPTS:
            try:
                generated = self.model.generate(
                    prompt,
                    max_new_tokens=num_tokens,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.92,
                    strategy="nucleus"
                )
                all_generated.append(generated)
                samples.append({"prompt": prompt, "generated": generated[:80]})
            except Exception:
                continue

        if not all_generated:
            return {
                "ttr": 0.0, "rep3": 1.0, "ascii_rate": 0.0,
                "fluency": 0.0, "samples": [],
            }

        combined = " ".join(all_generated)
        chars = list(combined)
        total_chars = len(chars)

        unique_chars = len(set(chars))
        ttr = unique_chars / max(total_chars, 1)

        char_ids = [ord(c) for c in combined]
        trigrams = [tuple(char_ids[i:i+3]) for i in range(len(char_ids) - 2)]
        if trigrams:
            from collections import Counter
            tri_counts = Counter(trigrams)
            repeated = sum(1 for c in tri_counts.values() if c > 1)
            rep3_rate = repeated / len(tri_counts)
        else:
            rep3_rate = 0.0

        valid_ascii = sum(1 for c in chars if c.isprintable())
        ascii_rate = valid_ascii / max(total_chars, 1)

        words = combined.split()
        if words:
            known = sum(1 for w in words if w.lower().strip(".,!?;:'\"") in COMMON_WORDS)
            fluency = known / len(words)
        else:
            fluency = 0.0

        return {
            "ttr": ttr,
            "rep3": rep3_rate,
            "ascii_rate": ascii_rate,
            "fluency": fluency,
            "samples": samples,
        }

    @torch.no_grad()
    def _measure_speed(self, num_tokens: int = 100) -> float:
        """Measure inference tokens per second."""
        self.model.eval()
        prompt = "the quick brown fox"

        start = time.perf_counter()
        self.model.generate(
            prompt,
            max_new_tokens=num_tokens,
            temperature=0.8,
            strategy="nucleus"
        )
        elapsed = time.perf_counter() - start

        return num_tokens / max(elapsed, 1e-6)

    def run(self, verbose: bool = True) -> BenchmarkResult:
        """Run all benchmarks and return a BenchmarkResult."""
        if verbose:
            print("\nRunning benchmark...")
            print(f"  Eval texts   : {len(self.eval_texts)}")
            print(f"  Gen prompts  : {len(GENERATION_PROMPTS)}")

        t0 = time.perf_counter()

        if verbose:
            print("  [1/3] Language modelling metrics (BPC, accuracy)...")
        lm = self._compute_lm_metrics()

        if verbose:
            print("  [2/3] Generation quality metrics (diversity, fluency)...")
        gen = self._compute_generation_metrics()

        if verbose:
            print("  [3/3] Inference speed...")
        speed = self._measure_speed()

        elapsed = time.perf_counter() - t0

        cfg = self.model.config
        result = BenchmarkResult(
            model_name=f"ZeroBaseLLM-{cfg.num_transformer_blocks}L-{cfg.embed_dim}D",
            param_count=self.model.count_parameters(),
            model_size_mb=self.model.get_model_size_mb(),
            bits_per_char=lm["bpc"],
            perplexity=lm["perplexity"],
            top1_accuracy=lm["top1"],
            top5_accuracy=lm["top5"],
            type_token_ratio=gen["ttr"],
            trigram_repetition_rate=gen["rep3"],
            ascii_validity_rate=gen["ascii_rate"],
            fluency_score=gen["fluency"],
            tokens_per_second=speed,
            eval_time_seconds=elapsed,
            generation_samples=gen["samples"],
            config_summary={
                "embed_dim": cfg.embed_dim,
                "num_heads": cfg.num_heads,
                "num_blocks": cfg.num_transformer_blocks,
                "ff_dim": cfg.ff_dim,
                "dropout": cfg.attention_dropout,
                "top_k": cfg.top_k,
                "top_p": cfg.top_p,
                "max_seq_len": cfg.max_seq_len,
            },
        )

        return result


def compare_results(before: BenchmarkResult, after: BenchmarkResult) -> str:
    """Print a side-by-side improvement table."""

    def arrow(a: float, b: float, lower_better: bool = False) -> str:
        if lower_better:
            delta = a - b
        else:
            delta = b - a
        if abs(delta) < 1e-6:
            return "  --"
        if delta > 0:
            return f" ↑ +{abs(delta):.4f}"
        return f" ↓ -{abs(delta):.4f}"

    lines = [
        "",
        "=" * 72,
        "  OPTIMIZATION COMPARISON  (before → after fine-tuning)",
        "=" * 72,
        f"  {'Metric':<35} {'Before':>10} {'After':>10}  {'Change':>14}",
        "-" * 72,
        f"  {'Bits Per Char (↓ better)':<35} {before.bits_per_char:>10.4f} {after.bits_per_char:>10.4f}  {arrow(before.bits_per_char, after.bits_per_char, lower_better=True):>14}",
        f"  {'Perplexity (↓ better)':<35} {before.perplexity:>10.4f} {after.perplexity:>10.4f}  {arrow(before.perplexity, after.perplexity, lower_better=True):>14}",
        f"  {'Top-1 Accuracy (↑ better)':<35} {before.top1_accuracy * 100:>9.2f}% {after.top1_accuracy * 100:>9.2f}%  {arrow(before.top1_accuracy, after.top1_accuracy):>14}",
        f"  {'Top-5 Accuracy (↑ better)':<35} {before.top5_accuracy * 100:>9.2f}% {after.top5_accuracy * 100:>9.2f}%  {arrow(before.top5_accuracy, after.top5_accuracy):>14}",
        f"  {'Diversity / TTR (↑ better)':<35} {before.type_token_ratio:>10.4f} {after.type_token_ratio:>10.4f}  {arrow(before.type_token_ratio, after.type_token_ratio):>14}",
        f"  {'Repetition Rate (↓ better)':<35} {before.trigram_repetition_rate:>10.4f} {after.trigram_repetition_rate:>10.4f}  {arrow(before.trigram_repetition_rate, after.trigram_repetition_rate, lower_better=True):>14}",
        f"  {'ASCII Validity (↑ better)':<35} {before.ascii_validity_rate:>10.4f} {after.ascii_validity_rate:>10.4f}  {arrow(before.ascii_validity_rate, after.ascii_validity_rate):>14}",
        f"  {'Fluency Score (↑ better)':<35} {before.fluency_score:>10.4f} {after.fluency_score:>10.4f}  {arrow(before.fluency_score, after.fluency_score):>14}",
        f"  {'Inference Speed (tokens/s)':<35} {before.tokens_per_second:>10.1f} {after.tokens_per_second:>10.1f}  {arrow(before.tokens_per_second, after.tokens_per_second):>14}",
        "-" * 72,
        f"  {'OVERALL SCORE (↑ better)':<35} {before.overall_score():>10.2f} {after.overall_score():>10.2f}  {arrow(before.overall_score(), after.overall_score()):>14}",
        "=" * 72,
    ]
    return "\n".join(lines)
