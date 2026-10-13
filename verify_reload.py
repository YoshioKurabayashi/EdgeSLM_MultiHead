"""Verify base and heads in a fresh Python process."""
import argparse
import json
from pathlib import Path

import torch

from data import read
from evaluate import evaluate_features
from model import MultiHead, assert_unchanged, fingerprint
from runtime import load_model, write_json
from test_generation import compare, snapshot
from tools import INTENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_v1"))
    args = parser.parse_args()
    meta = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    model_path = Path(meta["model_path"])
    base, processor = load_model(model_path)
    original = json.loads((args.run / "frozen_before.json").read_text(encoding="utf-8"))
    assert_unchanged(original, fingerprint(base))
    head = MultiHead(base, len(INTENTS), base.config.text_config.hidden_size, meta["head_kind"])
    baseline = torch.load(args.run / "generation_baseline.pt", weights_only=True, map_location="cpu")
    cache = torch.load(args.run / "features_test.pt", weights_only=True, map_location="cpu")
    if cache["metadata"] != meta:
        raise AssertionError("feature metadata mismatch")
    rows = read(args.data)["test"]
    if [r["id"] for r in rows] != cache["ids"]:
        raise AssertionError("test rows changed")
    reports = []
    for seed in meta["seeds"]:
        checkpoint_meta = head.load_heads(args.run / f"heads_seed{seed}.pt")
        if checkpoint_meta != meta:
            raise AssertionError("head metadata mismatch")
        measured, _ = evaluate_features(head, cache["features"], cache["labels"], rows)
        previous = json.loads((args.run / f"seed{seed}.json").read_text(encoding="utf-8"))
        if measured != previous["test"]:
            raise AssertionError("fresh-process metrics differ")
        reports.append({"seed": seed, "metrics_equal": True})
    generation = compare(baseline, snapshot(base, processor, "cuda"))
    report = {"frozen_weights_equal": True, "head_reload": reports, "generation": generation}
    write_json(args.run / "fresh_process_reload.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
