"""Text-only Japanese BERT baseline with trainable encoder and two heads."""
import argparse
import copy
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from data import read, render
from evaluate import metrics
from model import task_loss
from runtime import seed_everything, write_json
from tools import INTENTS

MODEL_ID = "cl-tohoku/bert-base-japanese-char-v3"
REVISION = "ccddee593a9e03a589ae1feedcd4d1eb06d6ba02"


class BertHeads(nn.Module):
    def __init__(self, snapshot):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(snapshot, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.intent = nn.Linear(hidden, len(INTENTS))
        self.guard = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return {"intent": self.intent(states), "guard": self.guard(states)}


def encode(tokenizer, rows):
    inputs = tokenizer([render(r) for r in rows], padding=True, truncation=False, return_tensors="pt")
    if inputs["input_ids"].shape[1] > 256:
        raise ValueError("BERT input exceeds 256 tokens; revise common input policy")
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}


@torch.no_grad()
def evaluate(model, encoded, rows, batch_size=16):
    model.eval()
    predictions = {"intent": [], "guard": []}
    for ids in torch.arange(len(rows)).split(batch_size):
        inputs = {k: v[ids].cuda() for k, v in encoded.items()}
        outputs = model(**inputs)
        for k in predictions:
            predictions[k].extend(outputs[k].argmax(-1).cpu().tolist())
    reports = {"intent": metrics([r["intent"] for r in rows], predictions["intent"], len(INTENTS)),
               "guard": metrics([r["guard"] for r in rows], predictions["guard"], 3)}
    reports["guard_by_stage"] = {
        stage: metrics([r["guard"] for r in rows if r["stage"] == stage],
                       [predictions["guard"][i] for i, r in enumerate(rows) if r["stage"] == stage], 3)
        for stage in ("input", "action", "output")}
    return reports, predictions


def fit(model, encoded, rows, epochs, seed, base_lr, head_lr):
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": base_lr},
        {"params": list(model.intent.parameters()) + list(model.guard.parameters()), "lr": head_lr},
    ], weight_decay=.01)
    best, best_epoch, best_state = -1., None, None
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.
        for indices in torch.randperm(len(rows["train"])).split(16):
            optimizer.zero_grad(set_to_none=True)
            inputs = {k: v[indices].cuda() for k, v in encoded["train"].items()}
            labels = {k: torch.tensor([rows["train"][i][k] for i in indices.tolist()], device="cuda")
                      for k in ("intent", "guard")}
            loss, _ = task_loss(model(**inputs), labels["intent"], labels["guard"])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite BERT loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices) / len(rows["train"])
        report, _ = evaluate(model, encoded["validation"], rows["validation"])
        score = (report["intent"]["macro_f1"] + report["guard"]["macro_f1"]) / 2
        history.append({"epoch": epoch + 1, "train_loss": total_loss, "validation_mean_macro_f1": score})
        if score > best:
            best, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"seed {seed} epoch {epoch + 1}: val_mean_f1={score:.3f}", flush=True)
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_validation_score": best, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_v1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--base-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output directory")
    args.out.mkdir(parents=True)
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    rows = read(args.data)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded = {split: encode(tokenizer, samples) for split, samples in rows.items()}
    metadata = {"model_id": MODEL_ID, "revision": REVISION, "snapshot": snapshot,
                "tokenizer": type(tokenizer).__name__, "seeds": args.seeds,
                "epochs": args.epochs, "base_lr": args.base_lr, "head_lr": args.head_lr,
                "data_hashes": json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))["splits"],
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "fugashi", "unidic-lite")},
                "note": "text-only BERT; no image QA, generation, or multimodal tool calls"}
    write_json(args.out / "metadata.json", metadata)
    results = []
    torch.cuda.reset_peak_memory_stats()
    for seed in args.seeds:
        seed_everything(seed)
        model = BertHeads(snapshot).cuda()
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        start = time.perf_counter()
        training = fit(model, encoded, rows, args.epochs, seed, args.base_lr, args.head_lr)
        test, prediction = evaluate(model, encoded["test"], rows["test"])
        checkpoint = args.out / f"model_seed{seed}.pt"
        torch.save({"state": model.state_dict(), "metadata": metadata}, checkpoint)
        result = {"seed": seed, "training": training, "test": test,
                  "test_predictions": prediction, "trainable_parameters": count,
                  "training_seconds": time.perf_counter() - start,
                  "checkpoint_bytes": checkpoint.stat().st_size}
        write_json(args.out / f"seed{seed}.json", result)
        print(f"seed {seed}: intent={test['intent']['accuracy']:.3f}, guard={test['guard']['accuracy']:.3f}", flush=True)
        results.append(result)
        del model
        torch.cuda.empty_cache()
    write_json(args.out / "summary.json", {"metadata": metadata,
               "trainable_parameters": results[0]["trainable_parameters"],
               "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
               "seeds": results})


if __name__ == "__main__":
    main()
