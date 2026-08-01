"""Remote zero-shot classification adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
DEFAULT_THRESHOLD = 0.8

type_dic = {
    "身份证": 0,
    "户口本": 1,
}


def classify(
    text: str,
    label_dict: Mapping[str, int],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    timeout: float = 20,
) -> int | None:
    """Return the best label id, or ``None`` when confidence is too low."""
    if not text.strip():
        raise ValueError("输入文本不能为空")
    if not label_dict:
        raise ValueError("标签字典不能为空")

    api_token = os.getenv("HF_API_TOKEN")
    if not api_token:
        raise RuntimeError("HF_API_TOKEN is required for remote classification")

    import requests

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_token}"},
        json={
            "inputs": text,
            "parameters": {"candidate_labels": list(label_dict)},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()

    try:
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Unexpected Hugging Face response: {result!r}") from exc

    return label_dict[top_label] if top_score >= threshold else None
