from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmptyingEvent:
    """Reward-relevant summary of one storage emptying event."""

    storage: str
    event_id: int
    mode: str
    count: int
    total_waiting_time: float

    @property
    def is_drain(self) -> bool:
        """Whether this was a full-storage drain event."""
        return self.mode == "Drain"

    @property
    def is_return(self) -> bool:
        """Whether this was a partial-storage return event."""
        return self.mode == "Return"