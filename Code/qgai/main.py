"""Application entry point."""

from __future__ import annotations

from .server import http, socket_process, socket_utils
from .server.console import log


def main() -> None:
    log("Starting the intelligent government service assistant")
    socket_process.run()
    socket_utils.run()
    http.run()


if __name__ == "__main__":
    main()
