from __future__ import annotations

import socket

from config.tcp_settings import TCPSettings

from .protocol import (
    Message,
    ProtocolError,
    message_from_json,
)


class ConnectionError(Exception):
    """Raised when a TCP connection operation fails."""


class TCPConnection:
    """
    Thin TCP connection wrapper using newline-delimited JSON messages.

    The connection is intentionally unaware of Plant Simulation,
    synchronization, RL, or handshake semantics.
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        encoding: str = "utf-8",
        recv_buffer_size: int = 4096,
        receive_timeout: float | None = None,
    ) -> None:
        self._socket = sock
        self._encoding = encoding
        self._recv_buffer_size = recv_buffer_size
        self._receive_buffer = bytearray()

        if receive_timeout is not None:
            if receive_timeout <= 0:
                raise ValueError(
                    "receive_timeout must be positive."
                )

            try:
                self._socket.settimeout(
                    receive_timeout
                )
            except OSError as exc:
                raise ConnectionError(
                    "Could not configure socket receive timeout."
                ) from exc

    @classmethod
    def connect(
        cls,
        settings: TCPSettings,
    ) -> TCPConnection:
        """Create a client connection using TCP settings."""
        try:
            sock = socket.create_connection(
                (
                    settings.host,
                    settings.port,
                ),
                timeout=settings.connect_timeout,
            )
        except OSError as exc:
            raise ConnectionError(
                f"Could not connect to "
                f"{settings.host}:{settings.port}."
            ) from exc

        try:
            sock.settimeout(
                settings.receive_timeout
            )
        except OSError as exc:
            sock.close()

            raise ConnectionError(
                "Could not configure socket receive timeout."
            ) from exc

        return cls(
            sock,
            receive_timeout=settings.receive_timeout,
        )

    @classmethod
    def from_socket(
        cls,
        sock: socket.socket,
        *,
        receive_timeout: float | None = None,
    ) -> TCPConnection:
        """Wrap an already-connected socket."""
        return cls(
            sock,
            receive_timeout=receive_timeout,
        )

    def send(
        self,
        message: Message,
    ) -> None:
        """
        Send one protocol message.

        Messages are framed using a trailing newline.
        """
        payload = (
            message.to_json().encode(
                self._encoding
            )
            + b"\n"
        )

        try:
            self._socket.sendall(
                payload
            )
        except OSError as exc:
            raise ConnectionError(
                "Failed to send message."
            ) from exc

    def receive(self) -> Message:
        """
        Receive exactly one newline-delimited protocol message.

        Blocks until a complete message is available or
        the peer closes the connection.
        """
        while True:
            newline_index = (
                self._receive_buffer.find(
                    b"\n"
                )
            )

            if newline_index >= 0:
                line = self._receive_buffer[
                    :newline_index
                ]

                del self._receive_buffer[
                    :newline_index + 1
                ]

                if not line.strip():
                    continue

                try:
                    payload = line.decode(
                        self._encoding
                    )
                except UnicodeDecodeError as exc:
                    raise ConnectionError(
                        "Received data is not valid UTF-8."
                    ) from exc

                try:
                    return message_from_json(
                        payload
                    )
                except ProtocolError as exc:
                    raise ConnectionError(
                        "Received malformed protocol message."
                    ) from exc

            try:
                chunk = self._socket.recv(
                    self._recv_buffer_size
                )
            except socket.timeout as exc:
                raise ConnectionError(
                    "Receive operation timed out."
                ) from exc
            except OSError as exc:
                raise ConnectionError(
                    "Failed to receive data."
                ) from exc

            if not chunk:
                raise ConnectionError(
                    "Connection closed before a complete "
                    "message was received."
                )

            self._receive_buffer.extend(
                chunk
            )

    def close(self) -> None:
        """Close the underlying socket."""
        try:
            self._socket.close()
        except OSError as exc:
            raise ConnectionError(
                "Failed to close connection."
            ) from exc

    @property
    def socket(self) -> socket.socket:
        """Return the underlying socket."""
        return self._socket