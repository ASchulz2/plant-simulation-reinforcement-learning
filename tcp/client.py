from __future__ import annotations

from config.tcp_settings import TCPSettings

from .connection import ConnectionError, TCPConnection
from .handshake import Handshake, HandshakeSnapshot
from .protocol import (
    AcknowledgementMessage,
    ActionMessage,
    HelloAckMessage,
    HelloMessage,
    Message,
    ObservationMessage,
    TerminalMessage,
    ResetMessage,
    ResetAcknowledgementMessage
)


class TCPClient:
    def __init__(
        self,
        connection: TCPConnection,
        *,
        settings: TCPSettings | None = None,
        protocol_version: str = "1.0",
    ) -> None:
        self._connection = connection
        self._settings = settings
        self._handshake = Handshake()
        self._protocol_version = protocol_version

    @classmethod
    def connect(
        cls,
        settings: TCPSettings,
        *,
        protocol_version: str = "1.0",
    ) -> TCPClient:
        connection = TCPConnection.connect(settings)

        return cls(
            connection,
            settings=settings,
            protocol_version=protocol_version,
        )

    def reconnect(self) -> None:
        """
        Replace the current TCP connection with a fresh connection.
        """
        if self._settings is None:
            raise RuntimeError(
                "Cannot reconnect TCPClient without TCP settings."
            )

        self._connection.close()
        self._connection = TCPConnection.connect(
            self._settings
        )
        self._handshake.reset()

    def send_hello(self) -> None:
        self._connection.send(
            HelloMessage(protocol_version=self._protocol_version)
        )

    def receive_hello_ack(self) -> HelloAckMessage:
        message = self._receive()

        if not isinstance(message, HelloAckMessage):
            raise ConnectionError(
                "Expected hello acknowledgement, "
                f"received {type(message).__name__}."
            )

        return message

    def receive_observation(self) -> ObservationMessage:
        message = self._receive()

        if not isinstance(message, ObservationMessage):
            raise ConnectionError(
                f"Expected observation, received {type(message).__name__}."
            )

        self._handshake.begin_decision(
            message.decision_point_id
        )

        return message

    def send_action(
        self,
        *,
        decision_point_id: int,
        action_id: int,
    ) -> None:
        self._handshake.send_action(
            decision_point_id,
            action_id,
        )

        self._connection.send(
            ActionMessage(
                decision_point_id=decision_point_id,
                action_id=action_id,
            )
        )

    def receive_acknowledgement(
        self,
    ) -> AcknowledgementMessage:
        message = self._receive()

        if not isinstance(message, AcknowledgementMessage):
            raise ConnectionError(
                "Expected acknowledgement, "
                f"received {type(message).__name__}."
            )

        self._handshake.receive_acknowledgement(
            message.decision_point_id
        )

        return message

    def complete_handshake(self) -> None:
        self._handshake.complete()

    def reset_handshake(self) -> None:
        """Reset the local transaction state at an episode boundary."""
        self._handshake.reset()

    def send_terminal(
        self,
        *,
        decision_point_id: int,
        reason: str,
    ) -> None:
        self._connection.send(
            TerminalMessage(
                decision_point_id=decision_point_id,
                reason=reason,
            )
        )

    def receive_terminal(self) -> TerminalMessage:
        message = self._receive()

        if not isinstance(message, TerminalMessage):
            raise ConnectionError(
                "Expected terminal message, "
                f"received {type(message).__name__}."
            )

        return message


    def receive_episode_event(
        self,
    ) -> ObservationMessage | TerminalMessage:
        message = self._receive()

        if isinstance(
            message,
            ObservationMessage,
        ):
            self._handshake.begin_decision(
                message.decision_point_id
            )
            return message

        if isinstance(
            message,
            TerminalMessage,
        ):
            return message

        raise ConnectionError(
            "Expected observation or terminal message, "
            f"received {type(message).__name__}."
        )

    def send_reset(
        self,
        *,
        episode_number: int,
    ) -> None:
        self._connection.send(
            ResetMessage(
                episode_number=episode_number
            )
        )


    def receive_reset_ack(
        self,
    ) -> ResetAcknowledgementMessage:

        message = self._receive()

        if not isinstance(
            message,
            ResetAcknowledgementMessage,
        ):
            raise ConnectionError(
                "Expected reset acknowledgement, "
                f"received {type(message).__name__}."
            )

        return message

    def receive_message(self) -> Message:
        return self._receive()


    def receive_episode_message(
        self,
    ) -> ObservationMessage | TerminalMessage:
        """
        Receive the next message belonging to the current episode.

        An observation starts a new decision transaction and therefore
        advances the local handshake state. A terminal message does not.
        """

        message = self._receive()

        if isinstance(message, ObservationMessage):
            self._handshake.begin_decision(
                message.decision_point_id
            )
            return message

        if isinstance(message, TerminalMessage):
            return message

        raise ConnectionError(
            "Expected observation or terminal message, "
            f"received {type(message).__name__}."
        )


    @property
    def handshake(self) -> HandshakeSnapshot:
        return self._handshake.snapshot

    def close(self) -> None:
        self._connection.close()

    def _receive(self) -> Message:
        return self._connection.receive()

    @property
    def can_reconnect(self) -> bool:
        return self._settings is not None