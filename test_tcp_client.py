from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tcp.client import TCPClient
from tcp.connection import ConnectionError
from tcp.handshake import HandshakeState
from tcp.protocol import (
    AcknowledgementMessage,
    ActionMessage,
    ObservationMessage,
    TerminalMessage,
)


def test_client_decision_handshake() -> None:
    connection = MagicMock()

    connection.receive.side_effect = [
        ObservationMessage(
            decision_point_id=7,
            state=None,
            terminal=False,
        ),
        AcknowledgementMessage(
            decision_point_id=7,
        ),
    ]



    client = TCPClient(connection)

    observation = client.receive_observation()

    assert observation.decision_point_id == 7
    assert client.handshake.state is HandshakeState.WAITING_FOR_ACTION

    client.send_action(
        decision_point_id=7,
        action_id=2,
    )

    assert client.handshake.state is HandshakeState.ACTION_SENT

    acknowledgement = (
        client.receive_acknowledgement()
    )

    assert acknowledgement.decision_point_id == 7
    assert client.handshake.state is HandshakeState.ACKNOWLEDGED

    client.complete_handshake()

    assert client.handshake.state is HandshakeState.IDLE

    sent_message = connection.send.call_args.args[0]

    assert isinstance(sent_message, ActionMessage)
    assert sent_message.decision_point_id == 7
    assert sent_message.action_id == 2


def test_client_receives_terminal() -> None:
    connection = MagicMock()

    connection.receive.return_value = TerminalMessage(
        decision_point_id=1538,
        reason="success",
    )

    client = TCPClient(connection)

    terminal = client.receive_terminal()

    assert isinstance(terminal, TerminalMessage)
    assert terminal.decision_point_id == 1538
    assert terminal.reason == "success"

    connection.receive.assert_called_once_with()


def test_client_rejects_non_terminal_message_when_receiving_terminal() -> None:
    connection = MagicMock()

    connection.receive.return_value = ObservationMessage(
        decision_point_id=1539,
        state=None,
        terminal=False,
    )

    client = TCPClient(connection)

    with pytest.raises(
        ConnectionError,
        match="Expected terminal message",
    ):
        client.receive_terminal()


def test_client_receive_episode_event_returns_observation() -> None:
    connection = MagicMock()

    observation = ObservationMessage(
        decision_point_id=12,
        state=None,
        terminal=False,
    )

    connection.receive.return_value = observation

    client = TCPClient(connection)

    event = client.receive_episode_event()

    assert event == observation
    assert isinstance(event, ObservationMessage)


def test_client_receive_episode_event_returns_terminal() -> None:
    connection = MagicMock()

    terminal = TerminalMessage(
        decision_point_id=1000,
        reason="success",
    )

    connection.receive.return_value = terminal

    client = TCPClient(connection)

    event = client.receive_episode_event()

    assert event == terminal
    assert isinstance(event, TerminalMessage)


def test_client_receive_episode_event_rejects_unexpected_message() -> None:
    connection = MagicMock()

    connection.receive.return_value = AcknowledgementMessage(
        decision_point_id=42,
    )

    client = TCPClient(connection)

    with pytest.raises(
        ConnectionError,
        match="Expected observation or terminal message",
    ):
        client.receive_episode_event()


def test_client_receives_episode_observation_and_updates_handshake() -> None:
    connection = MagicMock()

    connection.receive.return_value = ObservationMessage(
        decision_point_id=1,
        state=None,
        terminal=False,
    )

    client = TCPClient(connection)

    message = client.receive_episode_message()

    assert isinstance(message, ObservationMessage)
    assert message.decision_point_id == 1
    assert (
        client.handshake.state
        is HandshakeState.WAITING_FOR_ACTION
    )


def test_client_receives_episode_terminal_without_starting_decision() -> None:
    connection = MagicMock()

    connection.receive.return_value = TerminalMessage(
        decision_point_id=42,
        reason="success",
    )

    client = TCPClient(connection)

    message = client.receive_episode_message()

    assert isinstance(message, TerminalMessage)
    assert message.decision_point_id == 42
    assert message.reason == "success"
    assert (
        client.handshake.state
        is HandshakeState.IDLE
    )


def test_client_rejects_unexpected_episode_message() -> None:
    connection = MagicMock()

    connection.receive.return_value = ActionMessage(
        decision_point_id=1,
        action_id=0,
    )

    client = TCPClient(connection)

    with pytest.raises(
        ConnectionError,
        match="Expected observation or terminal message",
    ):
        client.receive_episode_message()

def test_client_receives_failure_terminal() -> None:
    connection = MagicMock()

    connection.receive.return_value = TerminalMessage(
        decision_point_id=42,
        reason="failure",
    )

    client = TCPClient(connection)

    message = client.receive_terminal()

    assert message.decision_point_id == 42
    assert message.reason == "failure"

def test_reset_handshake_exposes_handshake_reset():
    client = TCPClient(MagicMock())

    client._handshake.begin_decision(42)

    client.reset_handshake()

    snapshot = client.handshake

    assert snapshot.state is HandshakeState.IDLE
    assert snapshot.decision_point_id is None
    assert snapshot.action_id is None


def test_reset_handshake_returns_client_to_idle():
    client = TCPClient(MagicMock())

    client._handshake.begin_decision(42)
    client._handshake.send_action(
        decision_point_id=42,
        action_id=42,
    )

    client.reset_handshake()

    snapshot = client.handshake

    assert snapshot.state is HandshakeState.IDLE
    assert snapshot.decision_point_id is None
    assert snapshot.action_id is None


def test_reset_handshake_allows_new_observation_transaction():
    client = TCPClient(MagicMock())

    client._handshake.begin_decision(42)
    client._handshake.send_action(
        decision_point_id=42,
        action_id=42,
    )

    client.reset_handshake()

    client._handshake.begin_decision(1)

    snapshot = client.handshake

    assert snapshot.state is HandshakeState.WAITING_FOR_ACTION
    assert snapshot.decision_point_id == 1
    assert snapshot.action_id is None