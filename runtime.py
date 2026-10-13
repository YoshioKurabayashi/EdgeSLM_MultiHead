import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_model(model_path, device="cuda"):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=True, attn_implementation="eager")
    return base.to(device).eval(), processor


def encode_texts(processor, texts, device, max_length=256):
    conversations = [[{"role": "user", "content": text}] for text in texts]
    result = processor.apply_chat_template(conversations, tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False,
                                           padding=True, return_dict=True, return_tensors="pt")
    if result["input_ids"].shape[1] > max_length:
        raise ValueError("input exceeds max_length; cannot silently truncate context")
    return result.to(device)


def encode_image(processor, image, question, device):
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False, return_dict=True,
                                         return_tensors="pt").to(device)
