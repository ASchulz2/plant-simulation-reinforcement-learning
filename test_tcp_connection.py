from __future__ import annotations

import socket

import pytest

from tcp.connection import (
    ConnectionError,
    TCPConnection,
)
from tcp.protocol import HelloMessage


def test_send_and_receive_message() -> None:
    left, right = socket.socketpair()

    try:
        sender = TCPConnection.from_socket(left)
        receiver = TCPConnection.from_socket(right)

        message = HelloMessage(
            protocol_version="1.0"
        )

        sender.send(message)

        assert receiver.receive() == message

        sender.close()
        receiver.close()

    finally:
        left.close()
        right.close()


def test_multiple_messages() -> None:
    left, right = socket.socketpair()

    try:
        sender = TCPConnection.from_socket(left)
        receiver = TCPConnection.from_socket(right)

        first = HelloMessage(
            protocol_version="1.0"
        )

        second = HelloMessage(
            protocol_version="2.0"
        )

        sender.send(first)
        sender.send(second)

        assert receiver.receive() == first
        assert receiver.receive() == second

        sender.close()
        receiver.close()

    finally:
        left.close()
        right.close()


def test_receive_timeout_is_applied() -> None:
    left, right = socket.socketpair()

    try:
        connection = TCPConnection.from_socket(
            left,
            receive_timeout=12.5,
        )

        assert connection.socket.gettimeout() == 12.5

    finally:
        left.close()
        right.close()


def test_non_positive_receive_timeout_is_rejected() -> None:
    left, right = socket.socketpair()

    try:
        with pytest.raises(ValueError):
            TCPConnection.from_socket(
                left,
                receive_timeout=0.0,
            )

    finally:
        left.close()
        right.close()


def test_receive_timeout_is_enforced() -> None:
    left, right = socket.socketpair()

    try:
        connection = TCPConnection.from_socket(
            left,
            receive_timeout=0.05,
        )

        with pytest.raises(
            ConnectionError,
            match="Receive operation timed out",
        ):
            connection.receive()

    finally:
        left.close()
        right.close()