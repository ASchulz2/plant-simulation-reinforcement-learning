from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReturnEvent:
    """One material-unit return completed by ReturnBuffer."""

    event_id: int
    language: str
    waiting_time: float

    def __post_init__(self) -> None:
        if self.event_id < 1:
            raise ValueError("event_id must be positive")
        if not self.language:
            raise ValueError("language must be non-empty")
        if self.waiting_time < 0.0:
            raise ValueError("waiting_time must be non-negative")