"""WebSocket transport for interactive Q&A and face enrollment."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from .console import log
from .variable_pool import (
    error_response,
    socket_process_host,
    socket_process_port,
    user_dic,
    user_lock,
)


async def _qna_session(websocket: Any, user) -> None:
    while not user.tables_filler.is_finish:
        question_data = await user.inquire.async_activate()
        if question_data is None:
            user.tables_filler.next_table()
            continue
        key, question = question_data
        await websocket.send(question)
        answer = await websocket.recv()
        if not isinstance(answer, str):
            raise ValueError("Q&A answers must be text")
        value = await user.get_answer.async_activate(answer, key)
        user.tables_filler[key] = value
        user[key] = value
    await websocket.send(json.dumps({"type": "qna", "message": "success"}, ensure_ascii=False))


async def _face_training_session(websocket: Any, user, user_id: str) -> None:
    from .. import face

    images = []
    async for message in websocket:
        if not isinstance(message, bytes):
            await websocket.send(json.dumps({"type": "face_train", "message": "binary image required"}))
            continue
        extracted = face.face_fetcher(message)
        if extracted is not None:
            images.append(extracted)
        if len(images) >= 50:
            success = await user.train_face.async_activate(user_id, images)
            await websocket.send(
                json.dumps(
                    {"type": "face_train", "message": "success" if success else "failed"},
                    ensure_ascii=False,
                )
            )
            return
        await websocket.send(json.dumps({"type": "face_train", "message": "continue"}))


async def process_handler(websocket: Any) -> None:
    try:
        raw_handshake = await websocket.recv()
        if not isinstance(raw_handshake, str):
            raise ValueError("handshake must be JSON text")
        handshake = json.loads(raw_handshake)
        user_id = str(handshake["user_id"])
        request_hash = str(handshake.get("hash", ""))
        request_type = handshake["type"]

        with user_lock:
            user = user_dic.get(user_id)
        if user is None:
            await websocket.send(
                json.dumps(error_response(user_id, request_hash, "handshake is required"), ensure_ascii=False)
            )
            await websocket.close(code=4001, reason="handshake is required")
            return

        if request_type == "qna":
            await _qna_session(websocket, user)
        elif request_type == "face_train":
            await _face_training_session(websocket, user, user_id)
        else:
            await websocket.send(
                json.dumps(error_response(user_id, request_hash, "unknown socket type"), ensure_ascii=False)
            )
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        await websocket.send(json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False))
    finally:
        await websocket.close()


async def begin() -> None:
    import websockets

    async with websockets.serve(process_handler, socket_process_host, socket_process_port):
        log(f"process WebSocket started on {socket_process_host}:{socket_process_port}")
        await asyncio.Future()


def run() -> threading.Thread:
    thread = threading.Thread(
        target=asyncio.run,
        args=(begin(),),
        name="qgai-process-websocket",
        daemon=True,
    )
    thread.start()
    return thread
