"""Frozen multimodal Qwen with independent intent and guardrail heads."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


def last_valid(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden.shape[:2] != mask.shape or not mask.bool().any(dim=1).all():
        raise ValueError("invalid hidden states or attention mask")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
    return hidden[torch.arange(len(indices), device=mask.device), indices]


def fingerprint(module: nn.Module) -> dict[str, str]:
    result = {}
    for name, tensor in module.state_dict().items():
        bytes_view = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        result[name] = hashlib.sha256(bytes_view.tobytes()).hexdigest()
    return result


def assert_unchanged(before: dict, after: dict) -> None:
    changed = [k for k in before.keys() | after.keys() if before.get(k) != after.get(k)]
    if changed:
        raise AssertionError(f"Frozen model changed: {changed[:10]}")


class MultiHead(nn.Module):
    def __init__(self, base: nn.Module, intent_count: int, hidden_size: int,
                 head_kind: str = "linear", mlp_width: int = 256):
        super().__init__()
        if head_kind not in {"linear", "mlp"}:
            raise ValueError(head_kind)
        self.base = base.requires_grad_(False).eval()
        self.intent_count = intent_count
        self.hidden_size = hidden_size
        self.head_kind = head_kind
        self.mlp_width = mlp_width

        def make_head(classes):
            if head_kind == "linear":
                return nn.Linear(hidden_size, classes)
            return nn.Sequential(nn.Linear(hidden_size, mlp_width), nn.GELU(),
                                 nn.Linear(mlp_width, classes))

        device = next(base.parameters()).device
        self.heads = nn.ModuleDict({"intent": make_head(intent_count),
                                    "guard": make_head(3)}).to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def features(self, **processor_inputs):
        with torch.no_grad():
            outputs = self.base.model(**processor_inputs, use_cache=False)
            return last_valid(outputs.last_hidden_state, processor_inputs["attention_mask"]).float()

    def classify_features(self, features):
        return {name: head(features.float()) for name, head in self.heads.items()}

    def forward(self, **processor_inputs):
        return self.classify_features(self.features(**processor_inputs))

    def parameter_report(self):
        trainable = {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}
        if any(not name.startswith("heads.") for name in trainable):
            raise AssertionError("non-head parameter is trainable")
        return {"trainable": trainable, "trainable_total": sum(trainable.values()),
                "total": sum(p.numel() for p in self.parameters())}

    def save_heads(self, path, metadata):
        torch.save({"state": self.heads.state_dict(), "intent_count": self.intent_count,
                    "hidden_size": self.hidden_size, "head_kind": self.head_kind,
                    "mlp_width": self.mlp_width, "metadata": metadata}, path)

    def load_heads(self, path):
        data = torch.load(path, map_location=next(self.heads.parameters()).device, weights_only=True)
        for name in ("intent_count", "hidden_size", "head_kind", "mlp_width"):
            if data[name] != getattr(self, name):
                raise ValueError(f"head mismatch: {name}")
        self.heads.load_state_dict(data["state"])
        return data["metadata"]


def task_loss(logits, intent_labels, guard_labels):
    losses = {}
    for name, labels in (("intent", intent_labels), ("guard", guard_labels)):
        valid = labels != -100
        if valid.any():
            losses[name] = F.cross_entropy(logits[name][valid], labels[valid])
    if not losses:
        raise ValueError("batch has no supervised labels")
    return sum(losses.values()), losses
