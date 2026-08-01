"""Lazy face-recognition facade.

Model files are optional deployment assets, so importing :mod:`qgai` must not
attempt to load them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["cv2_predict", "cv2_train", "face_fetcher"]


def cv2_predict(*args: Any, **kwargs: Any):
    from .Predict import cv2_predict as implementation

    return implementation(*args, **kwargs)


def cv2_train(*args: Any, **kwargs: Any):
    from .Train import cv2_train as implementation

    return implementation(*args, **kwargs)


def face_fetcher(*args: Any, **kwargs: Any):
    from .face_fetcher import face_fetcher as implementation

    return implementation(*args, **kwargs)
