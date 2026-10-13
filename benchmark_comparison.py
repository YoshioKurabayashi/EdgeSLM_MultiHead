"""Single-input latency for MLP head and text-only BERT classifier."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from compare_bert import BertHeads
from compare_mlp import CachedMLP
from data import read, render
from runtime import write_json


def timed(fn, repeats=30, warmup=5, cuda=True):
    values = []
    with torch.no_grad():
        for index in range(repeats + warmup):
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            if cuda:
                torch.cuda.synchronize()
            if index >= warmup:
                values.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": statistics.median(values),
            "p95_ms": float(torch.tensor(values).quantile(.95)),
            "repeats": repeats, "warmup": warmup}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-run", type=Path, default=Path("runs/linear_v1"))
    parser.add_argument("--mlp-run", type=Path, default=Path("runs/mlp_v1"))
    parser.add_argument("--bert-run", type=Path, default=Path("runs/bert_v1"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    row = read(Path("data/synthetic_v1"))["test"][0]
    text = render(row)

    cache = torch.load(args.linear_run / "features_test.pt", map_location="cpu", weights_only=True)
    if cache["ids"][0] != row["id"]:
        raise AssertionError("Qwen feature/test row mismatch")
    mlp = CachedMLP(cache["features"].shape[-1]).cuda().eval()
    mlp.load_state_dict(torch.load(args.mlp_run / "heads_seed7.pt", weights_only=True)["state"])
    feature = cache["features"][0:1].cuda()
    mlp_time = timed(lambda: mlp.classify_features(feature))
    mlp_bytes = sum(p.numel() * p.element_size() for p in mlp.parameters())
    del mlp

    bert_meta = json.loads((args.bert_run / "metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(bert_meta["snapshot"], local_files_only=True)
    bert = BertHeads(bert_meta["snapshot"]).cuda().eval()
    bert.load_state_dict(torch.load(args.bert_run / "model_seed7.pt", weights_only=True)["state"])
    tokens = tokenizer(text, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in tokens.items() if k in {"input_ids", "attention_mask"}}
    torch.cuda.reset_peak_memory_stats()
    model_time = timed(lambda: bert(**inputs))
    tokenizer_time = timed(lambda: tokenizer(text, return_tensors="pt"), cuda=False)
    inference_peak_bytes = torch.cuda.max_memory_allocated()
    report = {"seed": 7, "same_test_row": row["id"], "batch": 1,
              "qwen_mlp_head_only": mlp_time, "qwen_mlp_head_parameter_bytes": mlp_bytes,
              "bert_model_only": model_time, "bert_tokenizer_only": tokenizer_time,
              "bert_input_tokens": inputs["input_ids"].shape[-1],
              "bert_inference_peak_allocated_bytes": inference_peak_bytes,
              "note": "Qwen model+Linear latency is in linear summary; MLP full Qwen pass not rerun. Single input on Windows GPU; no device deployment."}
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
