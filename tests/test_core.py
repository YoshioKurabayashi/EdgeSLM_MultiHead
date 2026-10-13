import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from data import audit, build, render
from evaluate import metrics
from model import MultiHead, assert_unchanged, fingerprint, last_valid, task_loss
from test_generation import compare
from tools import MockState, dispatch, extract_args, validate
from probe_tool_call import parse_tool_call


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(20, 8)

    def forward(self, input_ids, attention_mask, **_):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()
        self.lm_head = nn.Linear(8, 20, bias=False)


def test_padded_last_token_and_empty_input():
    values = torch.arange(8).reshape(2, 4, 1)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    assert last_valid(values, mask).flatten().tolist() == [3, 5]
    with pytest.raises(ValueError):
        last_valid(values, torch.zeros_like(mask))


def test_only_heads_train_and_all_base_tensors_remain_equal():
    torch.manual_seed(4)
    base = FakeQwen()
    initial = fingerprint(base)
    model = MultiHead(base, 6, 8)
    model.train()
    assert not base.training
    assert model.parameter_report()["trainable_total"] == 9 * 9
    ids = torch.tensor([[0, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    optimizer = torch.optim.SGD(model.heads.parameters(), lr=.2)
    old_head = fingerprint(model.heads)
    loss, separate = task_loss(model(input_ids=ids, attention_mask=mask),
                               torch.tensor([1, 2]), torch.tensor([0, 2]))
    assert set(separate) == {"intent", "guard"}
    loss.backward()
    assert all(p.grad is None for p in base.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.heads.parameters())
    optimizer.step()
    assert old_head != fingerprint(model.heads)
    assert_unchanged(initial, fingerprint(base))


def test_fingerprint_catches_single_changed_weight():
    base = FakeQwen()
    before = fingerprint(base)
    with torch.no_grad():
        next(base.parameters()).flatten()[0] += 1
    with pytest.raises(AssertionError, match="Frozen model changed"):
        assert_unchanged(before, fingerprint(base))


def test_partial_labels_only_update_relevant_head():
    model = MultiHead(FakeQwen(), 6, 8)
    logits = model.classify_features(torch.randn(3, 8))
    loss, _ = task_loss(logits, torch.tensor([1, -100, 2]), torch.tensor([-100, -100, -100]))
    loss.backward()
    assert all(p.grad is None for p in model.heads["guard"].parameters())
    assert all(p.grad is not None for p in model.heads["intent"].parameters())
    with pytest.raises(ValueError, match="no supervised"):
        task_loss(logits, torch.full((3,), -100), torch.full((3,), -100))


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_checkpoint_roundtrip(tmp_path, kind):
    base = FakeQwen()
    model = MultiHead(base, 6, 8, kind)
    inputs = torch.randn(3, 8)
    original = model.classify_features(inputs)
    path = tmp_path / "head.pt"
    model.save_heads(path, {"revision": "test"})
    restored = MultiHead(base, 6, 8, kind)
    assert restored.load_heads(path) == {"revision": "test"}
    after = restored.classify_features(inputs)
    assert all(torch.equal(original[k], after[k]) for k in original)


def test_generation_probe_detects_changed_tokens_or_logits():
    initial = [{"tokens": torch.tensor([1, 2]), "logits": torch.tensor([.1]), "text": "x"}]
    changed = copy.deepcopy(initial)
    changed[0]["tokens"][0] += 1
    with pytest.raises(AssertionError, match="token"):
        compare(initial, changed)
    changed = copy.deepcopy(initial)
    changed[0]["logits"][0] += .1
    with pytest.raises(AssertionError):
        compare(initial, changed)


def test_dataset_group_split_and_stage_labels():
    rows = build()
    assert {k: len(v) for k, v in rows.items()} == {"train": 172, "validation": 86, "test": 86}
    assert {r["intent"] for r in rows["train"] if r["intent"] != -100} == set(range(6))
    assert {r["guard"] for r in rows["train"]} == {0, 1, 2}
    assert {r["stage"] for r in rows["train"]} == {"input", "action", "output"}
    assert all("intent=" not in render(r) for records in rows.values() for r in records)
    bad = copy.deepcopy(rows)
    bad["test"][0]["group"] = rows["train"][0]["group"]
    with pytest.raises(ValueError, match="leakage"):
        audit(bad)


def test_policy_blocks_even_when_head_says_allow():
    state = MockState(motion="moving", permission="allowed")
    result = dispatch("OPEN_TRUNK", "トランクを開けて", "ALLOW", state)
    assert result["status"] == "BLOCK" and not state.calls
    state = MockState(motion="stopped", permission="denied")
    result = dispatch("SET_TEMPERATURE", "温度を24度にして", "ALLOW", state)
    assert result["status"] == "BLOCK" and state.temperature_c == 22 and not state.calls


def test_mock_tool_executes_with_validated_slot_and_rejects_missing_slot():
    state = MockState()
    result = dispatch("SET_TEMPERATURE", "温度を24度にして", "ALLOW", state)
    assert result["called"] and state.temperature_c == 24 and len(state.calls) == 1
    result = dispatch("SET_TEMPERATURE", "温度を変えて", "ALLOW", state)
    assert result["status"] == "REVIEW" and len(state.calls) == 1
    assert validate("SET_TEMPERATURE", {"temperature_c": 99}, state) == "BLOCK"
    assert extract_args("START_NAVIGATION", "東京駅まで案内して") == {"destination": "東京駅"}


def test_action_guard_catches_after_input_allow_before_execution():
    state = MockState()
    result = dispatch("SET_TEMPERATURE", "温度を24度にして", "ALLOW", state, "BLOCK")
    assert result["status"] == "BLOCK" and state.temperature_c == 22 and not state.calls


def test_guard_metric_counts_dangerous_false_allow():
    report = metrics([0, 0, 1, 2, 2], [0, 2, 1, 0, 2], 3)
    assert report["block_to_allow_count"] == 1
    assert report["block_to_allow_rate"] == .5
    assert report["allow_to_block_rate"] == .5


def test_official_tool_call_parser_rejects_missing_and_extra_arguments():
    valid = "<tool_call>\n<function=set_temperature>\n<parameter=temperature_c>\n24\n</parameter>\n</function>\n</tool_call>"
    assert parse_tool_call(valid) == {"name": "set_temperature", "args": {"temperature_c": 24}}
    assert parse_tool_call(valid.replace("24", "hot")) is None
    assert parse_tool_call(valid.replace("temperature_c", "unused")) is None
    assert parse_tool_call(valid + valid) is None


def test_official_tool_call_parser_accepts_zero_argument_tool():
    call = "<tool_call>\n<function=open_trunk>\n</function>\n</tool_call>"
    assert parse_tool_call(call) == {"name": "open_trunk", "args": {}}
