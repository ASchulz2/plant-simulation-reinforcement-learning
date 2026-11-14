from __future__ import annotations

from config.tcp_settings import TCPSettings
from tcp.server import TCPServer


def test_server_accepts_connection() -> None:
    import threading

    settings = TCPSettings(
        host="127.0.0.1",
        port=0,
    )

    server = TCPServer(settings)
    server.start()

    host, port = server.address

    import socket

    client = socket.create_connection(
        (host, port)
    )

    connection = server.accept()

    assert connection.socket is not None

    client.close()
    connection.close()
    server.close()


def test_server_reports_bound_port() -> None:
    settings = TCPSettings(
        host="127.0.0.1",
        port=0,
    )

    server = TCPServer(settings)
    server.start()

    host, port = server.address

    assert host == "127.0.0.1"
    assert port > 0

    server.close()