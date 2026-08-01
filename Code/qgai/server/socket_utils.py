"""WebSocket transport for face and speech utilities."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from .console import log
from .variable_pool import socket_utils_host, socket_utils_port


def _request_path(websocket: Any) -> str:
    request = getattr(websocket, "request", None)
    return getattr(request, "path", "")


async def _face_predict(websocket: Any) -> None:
    from .. import face

    predictions: dict[str, int] = {}
    required_samples = 50
    total = 0
    async for image in websocket:
        if not isinstance(image, bytes):
            continue
        extracted = face.face_fetcher(image)
        if extracted is None:
            continue
        predicted_id = face.cv2_predict([extracted], min_acc=0.5)
        if predicted_id is not None:
            predictions[predicted_id] = predictions.get(predicted_id, 0) + 1
        total += 1

        response = {"user_id": "", "type": "face_predict", "hash": "", "message": "continue"}
        if total >= required_samples and predictions:
            best_user, count = max(predictions.items(), key=lambda item: item[1])
            if count >= required_samples * 0.5:
                response.update(user_id=best_user, message="success")
        await websocket.send(json.dumps(response, ensure_ascii=False))
        if response["message"] == "success":
            return


async def utils_handle(websocket: Any) -> None:
    path = _request_path(websocket)
    if path == "/face_predict":
        await _face_predict(websocket)
        return

    if path == "/tts":
        from ..voice import text2voice

        async for text in websocket:
            if not isinstance(text, str):
                raise ValueError("TTS input must be text")
            await websocket.send(text2voice(text))
        return

    if path == "/stt":
        from ..voice import voice2text

        async for sound in websocket:
            if not isinstance(sound, bytes):
                raise ValueError("STT input must be binary audio")
            await websocket.send(voice2text(sound))
        return

    await websocket.send(json.dumps({"type": "error", "message": "unknown path"}))
    await websocket.close(code=4004, reason="unknown path")


async def begin() -> None:
    import websockets

    async with websockets.serve(utils_handle, socket_utils_host, socket_utils_port):
        log(f"utility WebSocket started on {socket_utils_host}:{socket_utils_port}")
        await asyncio.Future()


def run() -> threading.Thread:
    thread = threading.Thread(
        target=asyncio.run,
        args=(begin(),),
        name="qgai-utility-websocket",
        daemon=True,
    )
    thread.start()
    return thread
