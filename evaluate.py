"""Classification reports and safety-specific error rates."""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def metrics(truth, prediction, classes):
    pairs = [(int(a), int(b)) for a, b in zip(truth, prediction) if int(a) != -100]
    if not pairs:
        return {"count": 0, "accuracy": None, "macro_f1": None}
    y, p = map(list, zip(*pairs))
    result = {"count": len(y), "accuracy": float(accuracy_score(y, p)),
              "macro_f1": float(f1_score(y, p, labels=list(range(classes)), average="macro", zero_division=0)),
              "per_class": classification_report(y, p, labels=list(range(classes)),
                                                 output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(y, p, labels=list(range(classes))).tolist()}
    if classes == 3:
        blocks = [b for a, b in pairs if a == 2]
        allows = [b for a, b in pairs if a == 0]
        result.update({"block_count": len(blocks),
                       "block_to_allow_count": sum(b == 0 for b in blocks),
                       "block_to_allow_rate": sum(b == 0 for b in blocks) / len(blocks) if blocks else None,
                       "allow_to_block_rate": sum(b == 2 for b in allows) / len(allows) if allows else None})
    return result


def evaluate_features(model, features, labels, rows=None):
    model.eval()
    device = next(model.heads.parameters()).device
    with __import__("torch").no_grad():
        logits = model.classify_features(features.to(device))
    pred = {k: v.argmax(-1).cpu().tolist() for k, v in logits.items()}
    reports = {"intent": metrics(labels["intent"].tolist(), pred["intent"],
                                 logits["intent"].shape[-1]),
               "guard": metrics(labels["guard"].tolist(), pred["guard"], 3)}
    if rows is not None:
        reports["guard_by_stage"] = {
            stage: metrics([r["guard"] for r in rows if r["stage"] == stage],
                           [pred["guard"][i] for i, r in enumerate(rows) if r["stage"] == stage], 3)
            for stage in ("input", "action", "output")}
    return reports, pred
