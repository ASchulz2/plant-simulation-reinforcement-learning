from __future__ import annotations

import socket

from config.tcp_settings import TCPSettings

from .connection import (
    ConnectionError,
    TCPConnection,
)


class TCPServer:
    """Thin TCP listening-server wrapper."""

    def __init__(
        self,
        settings: TCPSettings,
    ) -> None:
        self._settings = settings
        self._socket: socket.socket | None = None

    def start(self) -> None:
        """Bind and start listening."""
        if self._socket is not None:
            return

        try:
            sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM,
            )

            sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1,
            )

            sock.bind(
                (
                    self._settings.host,
                    self._settings.port,
                )
            )

            sock.listen(
                self._settings.backlog
            )

        except OSError as exc:
            if "sock" in locals():
                sock.close()

            raise ConnectionError(
                "Could not start TCP server."
            ) from exc

        self._socket = sock

    def accept(self) -> TCPConnection:
        """Accept one connection."""
        if self._socket is None:
            raise ConnectionError(
                "TCP server has not been started."
            )

        try:
            connection, _ = self._socket.accept()
        except OSError as exc:
            raise ConnectionError(
                "Failed to accept TCP connection."
            ) from exc

        return TCPConnection.from_socket(
            connection,
            receive_timeout=(
                self._settings.receive_timeout
            ),
        )

    @property
    def address(self) -> tuple[str, int]:
        """Return the actual bound host and port."""
        if self._socket is None:
            raise ConnectionError(
                "TCP server has not been started."
            )

        host, port = self._socket.getsockname()
        return str(host), int(port)

    def close(self) -> None:
        """Close the listening socket."""
        if self._socket is None:
            return

        try:
            self._socket.close()
        except OSError as exc:
            raise ConnectionError(
                "Failed to close TCP server."
            ) from exc
        finally:
            self._socket = None