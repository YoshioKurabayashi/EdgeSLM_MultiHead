"""Fresh-process predictions from saved MLP and BERT checkpoints."""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from compare_bert import BertHeads, encode, evaluate
from compare_mlp import CachedMLP
from data import read
from evaluate import evaluate_features
from runtime import write_json


def main():
    rows = read(Path("data/synthetic_v1"))["test"]
    cache = torch.load("runs/linear_v1/features_test.pt", weights_only=True, map_location="cpu")
    if cache["ids"] != [r["id"] for r in rows]:
        raise AssertionError("test IDs differ")
    reports = {"mlp": [], "bert": []}
    for seed in (7, 17, 29):
        mlp = CachedMLP(cache["features"].shape[-1]).cuda().eval()
        mlp.load_state_dict(torch.load(f"runs/mlp_v1/heads_seed{seed}.pt", weights_only=True)["state"])
        report, _ = evaluate_features(mlp, cache["features"], cache["labels"], rows)
        saved = json.loads(Path(f"runs/mlp_v1/seed{seed}.json").read_text(encoding="utf-8"))["test"]
        if report != saved:
            raise AssertionError(f"MLP seed {seed} reload differs")
        reports["mlp"].append({"seed": seed, "metrics_equal": True})
        del mlp
    meta = json.loads(Path("runs/bert_v1/metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(meta["snapshot"], local_files_only=True)
    encoded = encode(tokenizer, rows)
    for seed in (7, 17, 29):
        bert = BertHeads(meta["snapshot"]).cuda().eval()
        bert.load_state_dict(torch.load(f"runs/bert_v1/model_seed{seed}.pt", weights_only=True)["state"])
        report, _ = evaluate(bert, encoded, rows)
        saved = json.loads(Path(f"runs/bert_v1/seed{seed}.json").read_text(encoding="utf-8"))["test"]
        if report != saved:
            raise AssertionError(f"BERT seed {seed} reload differs")
        reports["bert"].append({"seed": seed, "metrics_equal": True})
        del bert
    write_json("runs/comparison_reload.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
