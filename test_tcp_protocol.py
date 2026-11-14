from __future__ import annotations

from tcp.protocol import (
    ActionMessage,
    HelloAckMessage,
    ObservationMessage,
    ProtocolError,
    TerminalMessage,
    message_from_dict,
    message_from_json,
)


def test_observation_json_round_trip() -> None:
    message = ObservationMessage(
        decision_point_id=42,
        state=None,
        terminal=False,
    )

    encoded = message.to_json()
    decoded = message_from_json(encoded)

    assert decoded == message


def test_action_json_round_trip() -> None:
    message = ActionMessage(
        decision_point_id=42,
        action_id=2,
    )

    encoded = message.to_json()
    decoded = message_from_json(encoded)

    assert decoded == message


def test_hello_ack_json_round_trip() -> None:
    message = HelloAckMessage(
        protocol_version="1.0",
    )

    encoded = message.to_json()
    decoded = message_from_json(encoded)

    assert decoded == message


def test_terminal_json_round_trip() -> None:
    message = TerminalMessage(
        decision_point_id=1538,
        reason="success",
    )

    encoded = message.to_json()
    decoded = message_from_json(encoded)

    assert isinstance(decoded, TerminalMessage)
    assert decoded.decision_point_id == 1538
    assert decoded.reason == "success"


def test_terminal_message_preserves_terminal_id() -> None:
    message = TerminalMessage(
        decision_point_id=1000,
        reason="success",
    )

    decoded = message_from_dict(
        message.to_dict()
    )

    assert isinstance(decoded, TerminalMessage)
    assert decoded.decision_point_id == 1000
    assert decoded.reason == "success"


def test_terminal_message_supports_failure_reason() -> None:
    message = TerminalMessage(
        decision_point_id=731,
        reason="failure",
    )

    decoded = message_from_json(
        message.to_json()
    )

    assert isinstance(decoded, TerminalMessage)
    assert decoded.reason == "failure"


def test_unknown_message_type_is_rejected() -> None:
    data = {
        "type": "unknown",
    }

    try:
        message_from_dict(data)
    except ProtocolError:
        pass
    else:
        raise AssertionError(
            "Unknown message type was accepted."
        )