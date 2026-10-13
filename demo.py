"""Exercise the learned route, mock tool, dialogue and image QA."""
import argparse
import json
from pathlib import Path

import torch

from model import MultiHead
from runtime import encode_image, encode_texts, load_model, write_json
from test_generation import make_circle_image, make_image
from tools import GUARD, INTENTS, MockState, dispatch


@torch.no_grad()
def generate(base, processor, inputs, max_new_tokens=64):
    tokens = base.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                           pad_token_id=processor.tokenizer.pad_token_id)
    output = tokens[0, inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(output, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    base, processor = load_model(meta["model_path"])
    model = MultiHead(base, len(INTENTS), base.config.text_config.hidden_size, meta["head_kind"])
    model.load_heads(args.run / f"heads_seed{args.seed}.pt")
    examples = [
        {"utterance": "温度を24度にして", "state": MockState()},
        {"utterance": "トランクを開けて", "state": MockState(motion="moving")},
        {"utterance": "こんにちは", "state": MockState()},
    ]
    results = []
    for sample in examples:
        state = sample["state"]
        text = (f"判定段階: 入力時\n走行状態: {state.motion}。操作権限: {state.permission}。\n"
                f"ユーザー: {sample['utterance']}")
        inputs = encode_texts(processor, [text], "cuda")
        with torch.no_grad():
            logits = model(**inputs)
        intent = INTENTS[int(logits["intent"].argmax(-1))]
        guard = GUARD[int(logits["guard"].argmax(-1))]
        action_guard = None
        if intent not in {"NO_TOOL", "UNKNOWN"} and guard == "ALLOW":
            action_input = encode_texts(processor,
                [f"判定段階: 操作直前\n走行状態: {state.motion}。操作権限: {state.permission}。\n"
                 f"ユーザー: {sample['utterance']}\n候補操作: {intent}"], "cuda")
            with torch.no_grad():
                action_guard = GUARD[int(model(**action_input)["guard"].argmax(-1))]
        routed = dispatch(intent, sample["utterance"], guard, state, action_guard)
        # The tool result can be rendered without generation; model generation is a separate path.
        results.append({"utterance": sample["utterance"], "intent": intent, "input_guard": guard,
                        "action_guard": action_guard, "dispatch": routed, "mock_calls": state.calls})
    dialogue = generate(base, processor, encode_texts(processor, ["こんにちは。短く挨拶して。"], "cuda"))
    vision_red = generate(base, processor, encode_image(processor, make_image(),
                           "画像の色と形を日本語で答えて。", "cuda"))
    vision_blue = generate(base, processor, encode_image(processor, make_circle_image(),
                            "画像の色と形を日本語で答えて。", "cuda"))
    output_checks = {}
    for key, answer in (("dialogue", dialogue), ("red_square", vision_red), ("blue_circle", vision_blue)):
        output_input = encode_texts(processor, [f"判定段階: 応答後\n生成された回答: {answer}"], "cuda")
        with torch.no_grad():
            output_checks[key] = GUARD[int(model(**output_input)["guard"].argmax(-1))]
    report = {"pilot_routes": results, "dialogue": dialogue,
              "image_qa_red_square": vision_red, "image_qa_blue_circle": vision_blue,
              "output_guard": output_checks,
              "image_ground_truth": {"red": "白地に赤い四角", "blue": "白地に青い丸"},
              "note": "illustrative samples, not accuracy measurement; no real device called"}
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
