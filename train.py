"""Official Qwen checkpoint: frozen feature extraction and linear-head pilot."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import psutil
import torch

from data import read, render
from evaluate import evaluate_features
from model import MultiHead, assert_unchanged, fingerprint, task_loss
from runtime import encode_texts, load_model, seed_everything, write_json
from test_generation import compare, snapshot
from tools import INTENTS


def extract(model, processor, rows, batch_size, device):
    chunks = []
    for start in range(0, len(rows), batch_size):
        texts = [render(r) for r in rows[start:start + batch_size]]
        inputs = encode_texts(processor, texts, device)
        chunks.append(model.features(**inputs).cpu())
        if start % (batch_size * 8) == 0:
            print(f"  features {start}/{len(rows)}", flush=True)
    features = torch.cat(chunks)
    labels = {key: torch.tensor([r[key] for r in rows], dtype=torch.long)
              for key in ("intent", "guard")}
    return features, labels


def fit(model, train_data, validation_data, validation_rows, seed, epochs, lr):
    seed_everything(seed)
    device = next(model.heads.parameters()).device
    x, labels = train_data
    x = x.to(device)
    labels = {k: v.to(device) for k, v in labels.items()}
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=lr, weight_decay=.01)
    best, best_state, best_epoch = -1., None, None
    history = []
    for epoch in range(epochs):
        model.train()
        losses = {"intent": 0., "guard": 0.}
        for indices in torch.randperm(len(x), device=device).split(64):
            optimizer.zero_grad(set_to_none=True)
            total, separate = task_loss(model.classify_features(x[indices]),
                                        labels["intent"][indices], labels["guard"][indices])
            if not torch.isfinite(total):
                raise RuntimeError("non-finite loss")
            total.backward()
            optimizer.step()
            for name, value in separate.items():
                losses[name] += float(value.detach()) * len(indices) / len(x)
        report, _ = evaluate_features(model, *validation_data, validation_rows)
        score = (report["intent"]["macro_f1"] + report["guard"]["macro_f1"]) / 2
        history.append({"epoch": epoch + 1, "train_loss": losses,
                        "validation_mean_macro_f1": score})
        if score > best:
            best, best_epoch = score, epoch + 1
            best_state = copy.deepcopy(model.heads.state_dict())
    model.heads.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_validation_score": best, "history": history}


def benchmark(model, processor, device, text):
    inputs = encode_texts(processor, [text], device)
    features = model.features(**inputs)

    def timed(fn):
        values = []
        with torch.no_grad():
            for i in range(15):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                fn()
                if device == "cuda":
                    torch.cuda.synchronize()
                if i >= 5:
                    values.append((time.perf_counter() - start) * 1000)
        return {"p50_ms": statistics.median(values),
                "p95_ms": float(torch.tensor(values).quantile(.95)),
                "repetitions": len(values), "warmup": 5}

    return {"batch": 1, "input_tokens": inputs["input_ids"].shape[-1],
            "model_plus_heads": timed(lambda: model(**inputs)),
            "heads_only": timed(lambda: model.classify_features(features)),
            "note": "one synthetic input; tokenizer and guard re-passes excluded"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/qwen3_5_2b"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output directory to preserve earlier results")
    args.out.mkdir(parents=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = read(args.data)
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    metadata = {"model_id": "Qwen/Qwen3.5-2B", "model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                "model_path": str(args.model_path.resolve()), "model_config_sha256": hashlib.sha256((args.model_path / "config.json").read_bytes()).hexdigest(),
                "head_kind": args.head, "seeds": args.seeds, "epochs": args.epochs, "learning_rate": args.lr,
                "input_format": "chat_template_no_thinking_stage_state_utterance_last_valid_v1",
                "data": {split: hashlib.sha256((args.data / f"{split}.jsonl").read_bytes()).hexdigest() for split in rows},
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "scikit-learn", "pillow")}}
    write_json(args.out / "metadata.json", metadata)
    print("Loading frozen official Qwen checkpoint", flush=True)
    base, processor = load_model(args.model_path, args.device)
    print("Hashing all model tensors", flush=True)
    original = fingerprint(base)
    write_json(args.out / "frozen_before.json", original)
    baseline = snapshot(base, processor, args.device)
    torch.save(baseline, args.out / "generation_baseline.pt")
    model = MultiHead(base, len(INTENTS), config["text_config"]["hidden_size"], args.head)
    write_json(args.out / "parameters.json", model.parameter_report())
    attach_generation = compare(baseline, snapshot(base, processor, args.device))
    cache = {}
    start = time.perf_counter()
    for split, items in rows.items():
        print(f"Extracting {split}", flush=True)
        cache[split] = extract(model, processor, items, args.batch_size, args.device)
        torch.save({"features": cache[split][0], "labels": cache[split][1],
                    "ids": [r["id"] for r in items], "metadata": metadata}, args.out / f"features_{split}.pt")
    feature_seconds = time.perf_counter() - start
    results = []
    for seed in args.seeds:
        seed_everything(seed)
        model = MultiHead(base, len(INTENTS), config["text_config"]["hidden_size"], args.head)
        head_before = fingerprint(model.heads)
        start = time.perf_counter()
        training = fit(model, cache["train"], cache["validation"], rows["validation"],
                       seed, args.epochs, args.lr)
        if fingerprint(model.heads) == head_before:
            raise AssertionError("heads did not update")
        if any(p.grad is not None for p in base.parameters()):
            raise AssertionError("base accumulated gradients")
        assert_unchanged(original, fingerprint(base))
        checkpoint = args.out / f"heads_seed{seed}.pt"
        model.save_heads(checkpoint, metadata)
        test, prediction = evaluate_features(model, *cache["test"], rows["test"])
        model.load_heads(checkpoint)
        reloaded, _ = evaluate_features(model, *cache["test"], rows["test"])
        if test != reloaded:
            raise AssertionError("head reload changed metrics")
        generation = compare(baseline, snapshot(base, processor, args.device))
        result = {"seed": seed, "training": training, "test": test, "test_predictions": prediction,
                  "head_training_seconds": time.perf_counter() - start,
                  "head_checkpoint_bytes": checkpoint.stat().st_size,
                  "frozen_tensors_unchanged": True, "generation": generation}
        write_json(args.out / f"seed{seed}.json", result)
        print(f"seed {seed}: intent={test['intent']['accuracy']:.3f}, guard={test['guard']['accuracy']:.3f}", flush=True)
        results.append(result)
    summary = {"status": "synthetic_pilot_only", "metadata": metadata,
               "parameters": model.parameter_report(), "feature_seconds": feature_seconds,
               "attachment_generation": attach_generation,
               "benchmark": benchmark(model.eval(), processor, args.device, render(rows["test"][0])),
               "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
               "process_ram_bytes": psutil.Process().memory_info().rss,
               "seeds": results}
    write_json(args.out / "summary.json", summary)
    print(f"Done: {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
