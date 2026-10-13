"""Exact, bounded generation regression probes for frozen Qwen."""
import torch
from PIL import Image, ImageDraw

from runtime import encode_image, encode_texts


def make_image(color="red"):
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).rectangle((30, 30, 98, 98), fill=color)
    return image


def make_circle_image(color="blue"):
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).ellipse((30, 30, 98, 98), fill=color)
    return image


@torch.no_grad()
def snapshot(base, processor, device, max_new_tokens=12):
    base.eval()
    probes = [encode_texts(processor, ["こんにちは。短く挨拶してください。"], device),
              encode_texts(processor, ["温度を24度に設定してください。"], device),
              encode_image(processor, make_image(), "この画像の色と形は何ですか。", device)]
    result = []
    for inputs in probes:
        logits = base(**inputs, logits_to_keep=1, use_cache=False).logits.float().cpu()
        tokens = base.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                               pad_token_id=processor.tokenizer.pad_token_id)
        output = tokens[0, inputs["input_ids"].shape[-1]:].cpu()
        result.append({"logits": logits, "tokens": output,
                       "text": processor.tokenizer.decode(output, skip_special_tokens=True)})
    return result


def compare(before, after):
    if len(before) != len(after):
        raise AssertionError("probe count changed")
    for a, b in zip(before, after):
        if not torch.equal(a["tokens"], b["tokens"]):
            raise AssertionError("generation token IDs changed")
        torch.testing.assert_close(a["logits"], b["logits"], rtol=0, atol=0)
    return {"probe_count": len(before), "includes_image": True,
            "token_ids_equal": True, "logits_exactly_equal": True,
            "outputs": [r["text"] for r in after]}
