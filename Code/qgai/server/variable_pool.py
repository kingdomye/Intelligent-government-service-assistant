"""Shared in-memory server state."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from ..config import settings
from .cipher import AESCipher, ChiperBase
from .console import log

if TYPE_CHECKING:
    from ..user import User

http_host = settings.http_host
http_port = settings.http_port
socket_utils_host = settings.utils_ws_host
socket_utils_port = settings.utils_ws_port
socket_process_host = settings.process_ws_host
socket_process_port = settings.process_ws_port

user_dic: dict[str, "User"] = {}
user_lock = threading.RLock()
aes_cipher = AESCipher(settings.aes_key) if settings.aes_key else ChiperBase()


def processing_response(user_id: str, flow_hash: str, message: str = "") -> dict:
    log(f'"{user_id}" request is still processing')
    return {
        "user_id": user_id,
        "type": "processing",
        "hash": flow_hash,
        "message": message,
    }


def error_response(user_id: str, flow_hash: str, message: str = "") -> dict:
    log(f'"{user_id}" request failed: {message}')
    return {
        "user_id": user_id,
        "type": "error",
        "hash": flow_hash,
        "message": message,
    }
