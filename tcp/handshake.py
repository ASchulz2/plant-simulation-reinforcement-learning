from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HandshakeError(ValueError):
    """Raised when a handshake operation violates the protocol state."""


class HandshakeState(Enum):
    IDLE = "idle"
    WAITING_FOR_ACTION = "waiting_for_action"
    ACTION_SENT = "action_sent"
    ACKNOWLEDGED = "acknowledged"


@dataclass(frozen=True)
class HandshakeSnapshot:
    state: HandshakeState
    decision_point_id: int | None
    action_id: int | None


class Handshake:
    """Pure state machine for the Python side of the RL transaction."""

    def __init__(self) -> None:
        self._state = HandshakeState.IDLE
        self._decision_point_id: int | None = None
        self._action_id: int | None = None

    def begin_decision(self, decision_point_id: int) -> None:
        if self._state is not HandshakeState.IDLE:
            raise HandshakeError(
                "Cannot begin a new decision before the current "
                "transaction has completed."
            )

        self._state = HandshakeState.WAITING_FOR_ACTION
        self._decision_point_id = decision_point_id
        self._action_id = None

    def send_action(
        self,
        decision_point_id: int,
        action_id: int,
    ) -> None:
        if self._state is not HandshakeState.WAITING_FOR_ACTION:
            raise HandshakeError(
                "Action can only be sent after an observation."
            )

        self._validate_decision_point(decision_point_id)

        self._action_id = action_id
        self._state = HandshakeState.ACTION_SENT

    def receive_acknowledgement(self, decision_point_id: int) -> None:
        if self._state is not HandshakeState.ACTION_SENT:
            raise HandshakeError(
                "Acknowledgement can only be received after an action "
                "has been sent."
            )

        self._validate_decision_point(decision_point_id)

        self._state = HandshakeState.ACKNOWLEDGED

    def complete(self) -> None:
        if self._state is not HandshakeState.ACKNOWLEDGED:
            raise HandshakeError(
                "Handshake can only be completed after acknowledgement."
            )

        self.reset()

    def reset(self) -> None:
        """
        Reset the transaction state to IDLE.

        This operation is intended for an explicit episode or connection
        boundary where no further completion of the current transaction
        is expected.
        """

        self._state = HandshakeState.IDLE
        self._decision_point_id = None
        self._action_id = None

    def _validate_decision_point(self, decision_point_id: int) -> None:
        if decision_point_id != self._decision_point_id:
            raise HandshakeError(
                f"Wrong decision point: expected "
                f"{self._decision_point_id}, "
                f"received {decision_point_id}."
            )

    @property
    def snapshot(self) -> HandshakeSnapshot:
        return HandshakeSnapshot(
            state=self._state,
            decision_point_id=self._decision_point_id,
            action_id=self._action_id,
        )

    @property
    def state(self) -> HandshakeState:
        return self._state