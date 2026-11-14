from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, ClassVar


class ProtocolError(ValueError):
    """Raised when a TCP protocol message is malformed or unsupported."""


@dataclass(frozen=True)
class Message:
    """Base class for all protocol messages."""

    message_type: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the message into its wire-format dictionary."""
        data = asdict(self)
        data["type"] = self.message_type
        return data

    def to_json(self) -> str:
        """Serialize the message as a single JSON object."""
        return json.dumps(
            self.to_dict(),
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create a message from a dictionary."""
        raise NotImplementedError


@dataclass(frozen=True)
class HelloMessage(Message):
    """Initial protocol message."""

    message_type: ClassVar[str] = "hello"
    protocol_version: str = "1.0"

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> HelloMessage:
        return cls(
            protocol_version=data["protocol_version"],
        )


@dataclass(frozen=True)
class HelloAckMessage(Message):
    """Acknowledgement of the initial protocol hello."""

    message_type: ClassVar[str] = "hello_ack"
    protocol_version: str = "1.0"

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> HelloAckMessage:
        return cls(
            protocol_version=data["protocol_version"],
        )


@dataclass(frozen=True)
class ObservationMessage(Message):
    """Decision-point observation sent from Plant Simulation to Python."""

    message_type: ClassVar[str] = "observation"

    decision_point_id: int
    state: Any | None = None
    terminal: bool = False

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> ObservationMessage:
        return cls(
            decision_point_id=int(
                data["decision_point_id"]
            ),
            state=data.get("state"),
            terminal=bool(
                data.get("terminal", False)
            ),
        )


@dataclass(frozen=True)
class ActionMessage(Message):
    """Action sent from Python to Plant Simulation."""

    message_type: ClassVar[str] = "action"

    decision_point_id: int
    action_id: int

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> ActionMessage:
        return cls(
            decision_point_id=int(
                data["decision_point_id"]
            ),
            action_id=int(
                data["action_id"]
            ),
        )


@dataclass(frozen=True)
class AcknowledgementMessage(Message):
    """Acknowledgement sent by Python for a decision point."""

    message_type: ClassVar[str] = "ack"

    decision_point_id: int

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> AcknowledgementMessage:
        return cls(
            decision_point_id=int(
                data["decision_point_id"]
            ),
        )


@dataclass(frozen=True)
class TerminalMessage(Message):
    message_type: ClassVar[str] = "terminal"

    decision_point_id: int
    reason: str
    throughput: int | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> TerminalMessage:
        throughput = data.get("throughput")

        return cls(
            decision_point_id=int(
                data["decision_point_id"]
            ),
            reason=str(
                data["reason"]
            ),
            throughput=(
                int(throughput)
                if throughput is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ErrorMessage(Message):
    """Protocol-level error."""

    message_type: ClassVar[str] = "error"

    code: str
    message: str

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> ErrorMessage:
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
        )


@dataclass(frozen=True)
class ResetMessage(Message):
    message_type: ClassVar[str] = "reset"
    episode_number: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResetMessage:
        return cls(
            episode_number=int(data["episode_number"])
        )


@dataclass(frozen=True)
class ResetAcknowledgementMessage(Message):
    message_type: ClassVar[str] = "reset_ack"
    episode_number: int

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> ResetAcknowledgementMessage:
        return cls(
            episode_number=int(data["episode_number"])
        )


_MESSAGE_TYPES: dict[str, type[Message]] = {
    HelloMessage.message_type: HelloMessage,
    HelloAckMessage.message_type: HelloAckMessage,
    ObservationMessage.message_type: ObservationMessage,
    ActionMessage.message_type: ActionMessage,
    AcknowledgementMessage.message_type: AcknowledgementMessage,
    TerminalMessage.message_type: TerminalMessage,
    ErrorMessage.message_type: ErrorMessage,
    ResetMessage.message_type: ResetMessage,
    ResetAcknowledgementMessage.message_type: ResetAcknowledgementMessage
}


def message_from_dict(
    data: dict[str, Any],
) -> Message:
    """
    Deserialize a protocol message from a dictionary.

    The discriminator field ``type`` determines the concrete message type.
    """
    if not isinstance(data, dict):
        raise ProtocolError(
            "Message must be a JSON object."
        )

    message_type = data.get("type")

    if not isinstance(message_type, str):
        raise ProtocolError(
            "Message is missing a string 'type' field."
        )

    message_class = _MESSAGE_TYPES.get(
        message_type
    )

    if message_class is None:
        raise ProtocolError(
            f"Unknown message type: {message_type!r}"
        )

    try:
        return message_class.from_dict(data)
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ProtocolError(
            f"Malformed '{message_type}' message."
        ) from exc


def message_from_json(
    payload: str,
) -> Message:
    """Deserialize a protocol message from a JSON string."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            "Invalid JSON payload."
        ) from exc

    return message_from_dict(data)