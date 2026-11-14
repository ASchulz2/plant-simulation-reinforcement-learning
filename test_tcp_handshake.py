from __future__ import annotations

import pytest

from tcp.handshake import (
    Handshake,
    HandshakeError,
    HandshakeState,
)


def test_happy_path() -> None:
    handshake = Handshake()

    handshake.begin_decision(157)

    assert handshake.state is HandshakeState.WAITING_FOR_ACTION

    handshake.send_action(157, 2)

    assert handshake.state is HandshakeState.ACTION_SENT
    assert handshake.snapshot.action_id == 2

    handshake.receive_acknowledgement(157)

    assert handshake.state is HandshakeState.ACKNOWLEDGED

    handshake.complete()

    assert handshake.state is HandshakeState.IDLE
    assert handshake.snapshot.decision_point_id is None
    assert handshake.snapshot.action_id is None


def test_wrong_decision_point_when_sending_action_is_rejected() -> None:
    handshake = Handshake()
    handshake.begin_decision(157)

    with pytest.raises(HandshakeError):
        handshake.send_action(158, 2)


def test_wrong_acknowledgement_is_rejected() -> None:
    handshake = Handshake()
    handshake.begin_decision(157)
    handshake.send_action(157, 2)

    with pytest.raises(HandshakeError):
        handshake.receive_acknowledgement(158)


def test_action_before_decision_is_rejected() -> None:
    handshake = Handshake()

    with pytest.raises(HandshakeError):
        handshake.send_action(157, 2)


def test_second_decision_before_completion_is_rejected() -> None:
    handshake = Handshake()
    handshake.begin_decision(157)

    with pytest.raises(HandshakeError):
        handshake.begin_decision(158)


def test_acknowledgement_before_action_is_rejected() -> None:
    handshake = Handshake()
    handshake.begin_decision(157)

    with pytest.raises(HandshakeError):
        handshake.receive_acknowledgement(157)


def test_completion_requires_acknowledgement() -> None:
    handshake = Handshake()
    handshake.begin_decision(157)
    handshake.send_action(157, 2)

    with pytest.raises(HandshakeError):
        handshake.complete()


def test_new_decision_can_start_after_completion() -> None:
    handshake = Handshake()

    handshake.begin_decision(157)
    handshake.send_action(157, 2)
    handshake.receive_acknowledgement(157)
    handshake.complete()

    handshake.begin_decision(158)

    assert handshake.state is HandshakeState.WAITING_FOR_ACTION
    assert handshake.snapshot.decision_point_id == 158
    assert handshake.snapshot.action_id is None

def test_reset_returns_handshake_to_idle():
    handshake = Handshake()

    handshake.begin_decision(42)
    handshake.send_action(
        decision_point_id=42,
        action_id=42,
    )

    handshake.reset()

    snapshot = handshake.snapshot

    assert snapshot.state is HandshakeState.IDLE
    assert snapshot.decision_point_id is None
    assert snapshot.action_id is None


def test_reset_allows_new_decision_after_incomplete_transaction():
    handshake = Handshake()

    handshake.begin_decision(42)
    handshake.send_action(
        decision_point_id=42,
        action_id=42,
    )

    handshake.reset()

    handshake.begin_decision(1)

    assert handshake.state is HandshakeState.WAITING_FOR_ACTION
    assert handshake.snapshot.decision_point_id == 1
    assert handshake.snapshot.action_id is None


def test_complete_and_reset_produce_same_idle_state():
    completed = Handshake()
    completed.begin_decision(42)
    completed.send_action(
        decision_point_id=42,
        action_id=42,
    )
    completed.receive_acknowledgement(42)
    completed.complete()

    reset = Handshake()
    reset.begin_decision(42)
    reset.send_action(
        decision_point_id=42,
        action_id=42,
    )
    reset.reset()

    assert completed.snapshot == reset.snapshot
    assert completed.state is HandshakeState.IDLE