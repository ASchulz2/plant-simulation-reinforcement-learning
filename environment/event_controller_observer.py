from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EventControllerState(str, Enum):
    """Execution state derived from EventController signals."""

    RESET = "Reset"
    READY = "Ready"
    RUNNING = "Running"
    QUIESCENT = "Quiescent"


@dataclass(frozen=True)
class EventControllerObservation:
    """Snapshot of the relevant EventController values."""

    in_reset: bool
    is_running: bool
    sim_time: float
    state: EventControllerState


class EventControllerObserver:
    """Observe EventController execution state without accessing COM directly.

    The observer receives already-read EventController values and derives a
    small execution state machine from them.

    It deliberately does not know about Plant Simulation, COM, TCP, or RL
    episodes.
    """

    def __init__(self) -> None:
        self._state = EventControllerState.RESET

    @property
    def state(self) -> EventControllerState:
        """Return the most recently derived execution state."""
        return self._state

    def observe(
        self,
        *,
        in_reset: bool,
        is_running: bool,
        sim_time: float,
    ) -> EventControllerObservation:
        if not isinstance(in_reset, bool):
            raise TypeError("in_reset must be a bool")

        if not isinstance(is_running, bool):
            raise TypeError("is_running must be a bool")

        if not isinstance(sim_time, (int, float)):
            raise TypeError("sim_time must be numeric")

        if sim_time < 0.0:
            raise ValueError("sim_time must be non-negative")

        if in_reset:
            state = EventControllerState.RESET
        elif is_running:
            state = EventControllerState.RUNNING
        elif sim_time == 0.0:
            state = EventControllerState.READY
        else:
            state = EventControllerState.QUIESCENT

        self._state = state

        return EventControllerObservation(
            in_reset=in_reset,
            is_running=is_running,
            sim_time=float(sim_time),
            state=state,
        )