"""Lazy question-and-answer facade."""

from __future__ import annotations

from typing import Any

__all__ = ["inquire", "get_answer"]


def inquire(data: dict[str, Any]) -> dict[str, str] | None:
    """Return the first missing form field without loading the QA model."""
    for field, value in data.items():
        if value in (None, "", []):
            return {field: f"请问您的「{field}」是什么？"}
    return None


def get_answer(answer: str, key: str) -> str:
    from .Inquiry_and_GetAnswer import get_answer as implementation

    return implementation(answer, key)
