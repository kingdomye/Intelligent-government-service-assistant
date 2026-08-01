"""HTTP transport for user-session operations."""

from __future__ import annotations

import hashlib
import http.server
import json
from typing import Any

from ..user import AsyncModel, LoopBed, User
from .console import log
from .variable_pool import (
    aes_cipher,
    error_response,
    http_host,
    http_port,
    processing_response,
    user_dic,
    user_lock,
)

__all__ = ["Handler", "run"]


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "QGAI/1.1"

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        self._send_json({"status": "ok", "service": "qgai"})

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("request body is empty")
            encrypted_body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(aes_cipher.base64_de_str(encrypted_body))
            response = self.respond(request)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
            self._send_json({"type": "error", "message": str(exc)}, status=400)
            return
        except Exception as exc:
            log(f"unhandled HTTP request error: {exc}")
            self._send_json({"type": "error", "message": "internal server error"}, status=500)
            return

        encoded = aes_cipher.str_en_base64(json.dumps(response, ensure_ascii=False))
        body = encoded.encode("utf-8")
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        user_id = str(request["user_id"]).strip()
        request_type = str(request["type"]).strip()
        if not user_id:
            raise ValueError("user_id must not be empty")

        request_hash = request.get("hash") or hashlib.sha256(
            json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if request_type == "handshake":
            info = request.get("info")
            if not isinstance(info, dict):
                raise ValueError("handshake.info must be an object")
            with user_lock:
                if user_id not in user_dic:
                    user_dic[user_id] = User(info, LoopBed().looping_on_new_thread())
            return {
                "user_id": user_id,
                "type": "handshake",
                "hash": request_hash,
                "message": "success",
            }

        with user_lock:
            user = user_dic.get(user_id)
        if user is None:
            return error_response(user_id, request_hash, "handshake is required")

        if request_type == "classify":
            input_data = request.get("input")
            if not isinstance(input_data, dict) or not isinstance(input_data.get("text"), str):
                raise ValueError("classify.input.text must be a string")
            result = user.init(input_data["text"])
            if result == AsyncModel.processing:
                return processing_response(user_id, request_hash)
            return {
                "user_id": user_id,
                "type": "classify",
                "hash": request_hash,
                "classify": user.bus_type,
                "flow": user.flow,
            }

        if request_type == "storage":
            return {
                "user_id": user_id,
                "type": "storage",
                "hash": request_hash,
                "output": user.main_info,
            }

        if request_type == "summary":
            return {
                "user_id": user_id,
                "type": "summary",
                "hash": request_hash,
                "output": {
                    "tables": user.export_tables(),
                    "classify": user.bus_type,
                    "flow": user.flow,
                },
            }

        return error_response(user_id, request_hash, f"unknown request type: {request_type}")

    def log_message(self, message_format: str, *args: Any) -> None:
        log(message_format % args, "HTTP")


def run() -> None:
    address = (http_host, http_port)
    web = http.server.ThreadingHTTPServer(address, Handler)
    log(f"HTTP listener started on {http_host}:{http_port}")
    web.serve_forever()
