"""Lazy local-LLM flow generation.

Large model dependencies and weights are intentionally loaded on first use, not
when the application package is imported.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_runtime: tuple[Any, Any, Any] | None = None
_runtime_lock = threading.Lock()


def _load_runtime() -> tuple[Any, Any, Any]:
    global _runtime
    if _runtime is not None:
        return _runtime

    with _runtime_lock:
        if _runtime is not None:
            return _runtime

        base_model_path = os.getenv("QGAI_BASE_MODEL_PATH")
        lora_path = os.getenv("QGAI_LORA_MODEL_PATH")
        if not base_model_path or not lora_path:
            raise RuntimeError(
                "QGAI_BASE_MODEL_PATH and QGAI_LORA_MODEL_PATH are required "
                "for local flow generation"
            )

        from peft import PeftModel
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TextIteratorStreamer,
        )

        offload_dir = Path(os.getenv("QGAI_OFFLOAD_DIR", "offload_inference"))
        offload_dir.mkdir(parents=True, exist_ok=True)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=str(offload_dir),
            offload_state_dict=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            device_map="auto",
            offload_folder=str(offload_dir),
        )
        model.eval()
        _runtime = model, tokenizer, TextIteratorStreamer
        return _runtime


async def generate_streaming_response(user_input: dict, raw_text: str | None = None):
    model, tokenizer, streamer_class = _load_runtime()
    prompt = f"""
严格按以下规则处理：
1. 仅保留与用户特征相关的步骤；
2. 保留必要信息和详细流程；
3. 根据用户特征，使用便于用户理解的表达；
4. 如果用户数据不符合现实逻辑，输出“<-3>请输入正确的信息”，并附上原因；
5. 根据用户决定使用的语言；
6. 使用 Markdown，直接输出流程。

原始流程：{raw_text}
用户特征：{json.dumps(user_input, ensure_ascii=False)}
"""
    messages = [
        {
            "role": "system",
            "content": "你是简洁干练的政务流程精简专家，只输出规范的政务流程。",
        },
        {"role": "user", "content": prompt},
    ]
    payload = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        payload,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)
    streamer = streamer_class(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=10.0,
    )
    generate_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    }
    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=generate_kwargs,
        daemon=True,
    )
    generation_thread.start()
    for text in streamer:
        yield text
    generation_thread.join()
