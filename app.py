"""
Zero-Base LLM — Web Application

A public-facing Flask web app that lets anyone:
  - Chat with the model
  - Generate text from a prompt
  - View benchmark scores
  - Trigger a training round (self-improvement)
"""

import os
import json
import time
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

import torch
from zero_base_llm import ZeroBaseConfig, ZeroBaseLLM
from zero_base_llm.generation import TextGenerator, GenerationConfig

app = Flask(__name__)
CORS(app)

# ── Global model state ────────────────────────────────────────────────────────
MODEL_PATH = "checkpoints/best_model.pt"
BENCHMARK_PATH = "checkpoints/benchmark.json"

_model: ZeroBaseLLM = None
_generator: TextGenerator = None
_training = False
_training_progress = {"step": 0, "total": 0, "loss": None, "status": "idle"}
_lock = threading.Lock()


def _load_model() -> ZeroBaseLLM:
    """Load the best saved model or create a fresh one."""
    if Path(MODEL_PATH).exists():
        print(f"Loading model from {MODEL_PATH}")
        return ZeroBaseLLM.load(MODEL_PATH)
    print("No saved model found — creating fresh medium model")
    return ZeroBaseLLM(ZeroBaseConfig.medium())


def _get_model() -> ZeroBaseLLM:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _get_generator() -> TextGenerator:
    global _generator
    if _generator is None:
        model = _get_model()
        cfg = GenerationConfig(
            max_new_tokens=200,
            temperature=0.8,
            top_k=50,
            top_p=0.92,
            strategy="nucleus",
            repetition_penalty=1.2,
        )
        _generator = TextGenerator(model, cfg)
    return _generator


def _save_benchmark(result_dict: dict):
    Path("checkpoints").mkdir(exist_ok=True)
    with open(BENCHMARK_PATH, "w") as f:
        json.dump(result_dict, f, indent=2)


def _load_benchmark() -> dict:
    if Path(BENCHMARK_PATH).exists():
        with open(BENCHMARK_PATH) as f:
            return json.load(f)
    return {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/api/info")
def api_info():
    model = _get_model()
    benchmark = _load_benchmark()
    return jsonify({
        "parameters": model.count_parameters(),
        "size_mb": round(model.get_model_size_mb(), 2),
        "config": {
            "embed_dim": model.config.embed_dim,
            "num_heads": model.config.num_heads,
            "num_blocks": model.config.num_transformer_blocks,
            "ff_dim": model.config.ff_dim,
            "activation": getattr(model.config, "ff_activation", "gelu"),
            "dropout": model.config.attention_dropout,
        },
        "checkpoint": Path(MODEL_PATH).exists(),
        "benchmark": benchmark,
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    max_tokens = min(int(data.get("max_tokens", 150)), 400)
    temperature = float(data.get("temperature", 0.8))
    strategy = data.get("strategy", "nucleus")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        model = _get_model()
        gen_cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.92,
            strategy=strategy,
            repetition_penalty=1.2,
        )
        generator = TextGenerator(model, gen_cfg)
        t0 = time.perf_counter()
        output = generator.generate(prompt)
        elapsed = time.perf_counter() - t0
        clean = "".join(c if c.isprintable() else "" for c in output)
        new_text = clean[len(prompt):]
        return jsonify({
            "prompt": prompt,
            "generated": new_text,
            "full_text": clean,
            "tokens": len(new_text),
            "time_s": round(elapsed, 2),
            "tokens_per_sec": round(len(new_text) / max(elapsed, 0.001), 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    if _training:
        return jsonify({"error": "Training in progress — try again later"}), 409

    try:
        from zero_base_llm.benchmark import BenchmarkSuite
        model = _get_model()
        suite = BenchmarkSuite(model)
        result = suite.run(verbose=False)
        result_dict = {
            "model_name": result.model_name,
            "param_count": result.param_count,
            "size_mb": result.model_size_mb,
            "bits_per_char": round(result.bits_per_char, 4),
            "perplexity": round(result.perplexity, 4),
            "top1_accuracy": round(result.top1_accuracy * 100, 2),
            "top5_accuracy": round(result.top5_accuracy * 100, 2),
            "diversity_ttr": round(result.type_token_ratio, 4),
            "repetition_rate": round(result.trigram_repetition_rate, 4),
            "ascii_validity": round(result.ascii_validity_rate, 4),
            "fluency_score": round(result.fluency_score, 4),
            "tokens_per_sec": round(result.tokens_per_second, 1),
            "overall_score": result.overall_score(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_benchmark(result_dict)
        return jsonify(result_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    global _training, _training_progress, _model, _generator

    if _training:
        return jsonify({"error": "Training already in progress"}), 409

    data = request.get_json(force=True)
    steps = min(int(data.get("steps", 500)), 5000)

    def _run_training():
        global _training, _training_progress, _model, _generator
        try:
            _training = True
            _training_progress = {"step": 0, "total": steps, "loss": None, "status": "starting"}

            from zero_base_llm.training import SelfStudyTrainer
            from zero_base_llm.training.trainer import DIVERSE_SEED_TEXTS

            model = _get_model()
            trainer = SelfStudyTrainer(model, seed_texts=DIVERSE_SEED_TEXTS)
            train_iter = iter(trainer.train_loader)
            import math

            def lr_lambda(step):
                ws = trainer.warmup_steps
                if step < ws:
                    return (step + 1) / max(ws, 1)
                progress = (step - ws) / max(steps - ws, 1)
                return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

            for g in trainer.optimizer.param_groups:
                g["lr"] = model.config.learning_rate
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
                trainer.optimizer, lr_lambda
            )

            _training_progress["status"] = "training"
            for step in range(steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(trainer.train_loader)
                    batch = next(train_iter)

                metrics = trainer.train_step(batch)
                _training_progress["step"] = step + 1
                _training_progress["loss"] = round(metrics["loss"], 4)
                _training_progress["lr"] = round(metrics["lr"], 6)

            _training_progress["status"] = "saving"
            Path("checkpoints").mkdir(exist_ok=True)
            model.save(MODEL_PATH)

            _training_progress["status"] = "benchmarking"
            from zero_base_llm.benchmark import BenchmarkSuite
            suite = BenchmarkSuite(model)
            result = suite.run(verbose=False)
            result_dict = {
                "model_name": result.model_name,
                "param_count": result.param_count,
                "size_mb": result.model_size_mb,
                "bits_per_char": round(result.bits_per_char, 4),
                "perplexity": round(result.perplexity, 4),
                "top1_accuracy": round(result.top1_accuracy * 100, 2),
                "top5_accuracy": round(result.top5_accuracy * 100, 2),
                "diversity_ttr": round(result.type_token_ratio, 4),
                "repetition_rate": round(result.trigram_repetition_rate, 4),
                "ascii_validity": round(result.ascii_validity_rate, 4),
                "fluency_score": round(result.fluency_score, 4),
                "tokens_per_sec": round(result.tokens_per_second, 1),
                "overall_score": result.overall_score(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _save_benchmark(result_dict)

            with _lock:
                _model = model
                _generator = None

            _training_progress["status"] = "done"
            _training_progress["benchmark"] = result_dict

        except Exception as e:
            _training_progress["status"] = f"error: {e}"
        finally:
            _training = False

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    return jsonify({"message": f"Training started for {steps} steps", "steps": steps})


@app.route("/api/train/status")
def api_train_status():
    return jsonify({**_training_progress, "running": _training})


if __name__ == "__main__":
    print("Starting Zero-Base LLM Web App...")
    print("Pre-loading model...")
    _get_model()
    print("Model ready!")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
