from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Deque

from .return_event import ReturnEvent


class ReturnEventQueue:
    """FIFO queue for completed ReturnBuffer events.

    The queue is deliberately independent of Plant Simulation and TCP.  It
    only stores semantic :class:`ReturnEvent` objects that have already been
    created by an event reader or adapter.
    """

    def __init__(self) -> None:
        self._events: Deque[ReturnEvent] = deque()

    @property
    def size(self) -> int:
        """Return the number of events currently waiting in the queue."""
        return len(self._events)

    @property
    def empty(self) -> bool:
        """Return whether the queue contains no events."""
        return not self._events

    def put(self, event: ReturnEvent) -> None:
        """Append one return event to the queue."""
        if not isinstance(event, ReturnEvent):
            raise TypeError(
                "ReturnEventQueue accepts ReturnEvent instances only."
            )

        self._events.append(event)

    def put_many(self, events: Iterable[ReturnEvent]) -> None:
        """Append multiple return events while preserving their order."""
        for event in events:
            self.put(event)

    def get(self) -> ReturnEvent:
        """Remove and return the oldest queued event.

        Raises:
            IndexError: If the queue is empty.
        """
        return self._events.popleft()

    def drain(self) -> tuple[ReturnEvent, ...]:
        """Remove and return all queued events in FIFO order."""
        events = tuple(self._events)
        self._events.clear()
        return events
