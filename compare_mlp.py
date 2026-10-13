"""MLP heads on exactly the frozen Qwen features used by the linear run."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from torch import nn

from data import read
from evaluate import evaluate_features
from runtime import seed_everything, write_json
from tools import INTENTS
from train import fit


class CachedMLP(nn.Module):
    def __init__(self, hidden_size=2048, width=256):
        super().__init__()
        def head(classes):
            return nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(),
                                 nn.Linear(width, classes))
        self.heads = nn.ModuleDict({"intent": head(len(INTENTS)), "guard": head(3)})

    def classify_features(self, features):
        return {name: head(features.float()) for name, head in self.heads.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output directory")
    args.out.mkdir(parents=True)
    rows = read(args.data)
    linear_meta = json.loads((args.linear_run / "metadata.json").read_text(encoding="utf-8"))
    cache = {}
    for split in rows:
        cache[split] = torch.load(args.linear_run / f"features_{split}.pt", weights_only=True, map_location="cpu")
        if cache[split]["metadata"] != linear_meta or cache[split]["ids"] != [r["id"] for r in rows[split]]:
            raise AssertionError("cached features or dataset mismatch")
    metadata = {"kind": "Frozen Qwen + MLP Head", "source_run": str(args.linear_run.resolve()),
                "source_model_revision": linear_meta["model_revision"], "source_data_hashes": linear_meta["data"],
                "seeds": args.seeds, "epochs": args.epochs, "lr": args.lr,
                "torch": importlib.metadata.version("torch")}
    write_json(args.out / "metadata.json", metadata)
    results = []
    for seed in args.seeds:
        seed_everything(seed)
        model = CachedMLP(cache["train"]["features"].shape[1]).cuda()
        count = sum(p.numel() for p in model.parameters())
        start = time.perf_counter()
        training = fit(model,
                       (cache["train"]["features"], cache["train"]["labels"]),
                       (cache["validation"]["features"], cache["validation"]["labels"]),
                       rows["validation"], seed, args.epochs, args.lr)
        test, prediction = evaluate_features(model, cache["test"]["features"],
                                             cache["test"]["labels"], rows["test"])
        checkpoint = args.out / f"heads_seed{seed}.pt"
        torch.save({"state": model.state_dict(), "metadata": metadata}, checkpoint)
        result = {"seed": seed, "training": training, "test": test,
                  "test_predictions": prediction, "trainable_parameters": count,
                  "train_seconds": time.perf_counter() - start,
                  "checkpoint_bytes": checkpoint.stat().st_size}
        write_json(args.out / f"seed{seed}.json", result)
        print(f"seed {seed}: intent={test['intent']['accuracy']:.3f}, guard={test['guard']['accuracy']:.3f}", flush=True)
        results.append(result)
    write_json(args.out / "summary.json", {"metadata": metadata,
               "trainable_parameters": results[0]["trainable_parameters"], "seeds": results,
               "note": "same cached text features as Linear; no image-head training"})


if __name__ == "__main__":
    main()
