"""Speech conversion helpers with lazy model initialization."""

from __future__ import annotations

import io
import os
import tempfile
from functools import lru_cache


def bin2pcm(binary: bytes, code_type: str = "webm"):
    import librosa
    import numpy as np

    if code_type == "wav":
        pcm, _ = librosa.load(io.BytesIO(binary), sr=16000, mono=True)
        return pcm
    if code_type == "webm":
        import ffmpeg

        pcm_bytes, _ = (
            ffmpeg.input("pipe:0", format="webm")
            .output("pipe:1", format="f32le", ac=1, ar=16000)
            .run(input=binary, capture_stdout=True, quiet=True)
        )
        return np.frombuffer(pcm_bytes, dtype=np.float32)
    raise ValueError(f"unsupported audio type: {code_type}")


def wav2webm(wav_binary: bytes) -> bytes:
    import ffmpeg

    result, _ = (
        ffmpeg.input("pipe:0", format="wav")
        .output("pipe:1", format="webm")
        .run(input=wav_binary, capture_stdout=True, quiet=True)
    )
    return result


@lru_cache(maxsize=1)
def _speech_to_text_runtime():
    import whisper
    from opencc import OpenCC

    model_name_or_path = os.getenv("QGAI_WHISPER_MODEL", "medium")
    return whisper.load_model(model_name_or_path), OpenCC("t2s")


def voice2text(webm_binary: bytes) -> str:
    import torch

    model, converter = _speech_to_text_runtime()
    pcm = bin2pcm(webm_binary, "webm")
    result = model.transcribe(audio=torch.tensor(pcm, dtype=torch.float32))
    return converter.convert(result["text"])


def text2voice(text: str) -> bytes:
    import pyttsx3

    with tempfile.TemporaryDirectory(prefix="qgai-tts-") as temp_dir:
        wav_path = os.path.join(temp_dir, "speech.wav")
        engine = pyttsx3.init()
        engine.setProperty("rate", 200)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        with open(wav_path, "rb") as wav_file:
            return wav2webm(wav_file.read())
