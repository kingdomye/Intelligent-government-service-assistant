"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _port(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    http_host: str
    http_port: int
    process_ws_host: str
    process_ws_port: int
    utils_ws_host: str
    utils_ws_port: int
    aes_key: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            http_host=os.getenv("QGAI_HTTP_HOST", "127.0.0.1"),
            http_port=_port("QGAI_HTTP_PORT", 10925),
            process_ws_host=os.getenv("QGAI_PROCESS_WS_HOST", "127.0.0.1"),
            process_ws_port=_port("QGAI_PROCESS_WS_PORT", 4440),
            utils_ws_host=os.getenv("QGAI_UTILS_WS_HOST", "127.0.0.1"),
            utils_ws_port=_port("QGAI_UTILS_WS_PORT", 3304),
            aes_key=os.getenv("QGAI_AES_KEY") or None,
        )


settings = Settings.from_env()
