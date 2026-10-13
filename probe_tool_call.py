"""Read-only Qwen native tool-call probe using the official chat template."""
import argparse
import json
import re
from pathlib import Path

import torch

from runtime import load_model, write_json
from tools import MockState, validate

TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": "set_temperature",
      "description": "Set the mock cabin temperature in degrees Celsius",
      "parameters": {"type": "object", "properties": {"temperature_c": {"type": "integer"}},
                     "required": ["temperature_c"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "set_volume",
      "description": "Set the mock audio volume from zero to ten",
      "parameters": {"type": "object", "properties": {"volume": {"type": "integer"}},
                     "required": ["volume"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "open_trunk",
      "description": "Open the mock trunk",
      "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "start_navigation",
      "description": "Start mock navigation to a destination",
      "parameters": {"type": "object", "properties": {"destination": {"type": "string"}},
                     "required": ["destination"], "additionalProperties": False}}},
]

PROBES = [
    ("温度を24度にして", "set_temperature", {"temperature_c": 24}),
    ("音量を3にして", "set_volume", {"volume": 3}),
    ("トランクを開けて", "open_trunk", {}),
    ("東京駅まで案内して", "start_navigation", {"destination": "東京駅"}),
    ("こんにちは", None, None),
]


def parse_tool_call(raw):
    matches = re.findall(r"<tool_call>\s*<function=([a-z_]+)>(.*?)</function>\s*</tool_call>",
                         raw, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    name, content = matches[0]
    allowed = {t["function"]["name"]: t["function"]["parameters"] for t in TOOL_DEFINITIONS}
    if name not in allowed:
        return None
    parameters = re.findall(r"<parameter=([a-z_]+)>\s*(.*?)\s*</parameter>", content, flags=re.DOTALL)
    if len(parameters) != content.count("<parameter=") or len({k for k, _ in parameters}) != len(parameters):
        return None
    args = {}
    for key, value in parameters:
        if key not in allowed[name]["properties"]:
            return None
        expected = allowed[name]["properties"][key]["type"]
        if expected == "integer":
            if not re.fullmatch(r"[+-]?\d+", value.strip()):
                return None
            args[key] = int(value.strip())
        else:
            args[key] = value.strip()
    if set(args) != set(allowed[name].get("required", [])):
        return None
    return {"name": name, "args": args}


@torch.no_grad()
def run(model_path):
    base, processor = load_model(model_path)
    records = []
    for utterance, expected_name, expected_args in PROBES:
        messages = [{"role": "user", "content": utterance}]
        inputs = processor.apply_chat_template(messages, tools=TOOL_DEFINITIONS,
                                               tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False, return_dict=True,
                                               return_tensors="pt").to("cuda")
        tokens = base.generate(**inputs, do_sample=False, max_new_tokens=96,
                               pad_token_id=processor.tokenizer.pad_token_id)
        raw = processor.tokenizer.decode(tokens[0, inputs["input_ids"].shape[-1]:],
                                         skip_special_tokens=True)
        parsed = parse_tool_call(raw)
        match = ((parsed is None and expected_name is None) or
                 (parsed is not None and parsed["name"] == expected_name and parsed["args"] == expected_args))
        cpu_policy = (validate(parsed["name"].upper(), parsed["args"], MockState())
                      if parsed else None)
        records.append({"utterance": utterance, "expected_tool": expected_name,
                        "expected_args": expected_args, "raw": raw, "parsed": parsed,
                        "exact_match": match, "cpu_policy": cpu_policy})
        print(json.dumps({"utterance": utterance, "exact_match": match, "parsed": parsed}, ensure_ascii=False), flush=True)
    return {"model": str(model_path), "count": len(records),
            "exact_match_rate": sum(x["exact_match"] for x in records) / len(records),
            "records": records, "note": "five synthetic probes only; model did not execute any tool"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/qwen3_5_2b"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, run(args.model_path))


if __name__ == "__main__":
    main()
