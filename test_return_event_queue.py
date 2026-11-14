from __future__ import annotations

import pytest

from environment.return_event import ReturnEvent
from environment.return_event_queue import ReturnEventQueue


def make_event(
    event_id: int,
    language: str = "FR",
    waiting_time: float = 36.0,
) -> ReturnEvent:
    return ReturnEvent(
        event_id=event_id,
        language=language,
        waiting_time=waiting_time,
    )


def test_new_queue_is_empty() -> None:
    queue = ReturnEventQueue()

    assert queue.empty is True
    assert queue.size == 0


def test_put_and_get_preserve_single_event() -> None:
    queue = ReturnEventQueue()
    event = make_event(1)

    queue.put(event)

    assert queue.empty is False
    assert queue.size == 1
    assert queue.get() == event
    assert queue.empty is True
    assert queue.size == 0


def test_multiple_events_are_returned_in_fifo_order() -> None:
    queue = ReturnEventQueue()
    events = (
        make_event(1, "GER", 40.0),
        make_event(2, "FR", 36.0),
        make_event(3, "RUS", 42.5),
    )

    queue.put_many(events)

    assert queue.size == 3
    assert queue.drain() == events
    assert queue.empty is True


def test_drain_consumes_all_events() -> None:
    queue = ReturnEventQueue()
    queue.put_many(
        (
            make_event(1),
            make_event(2),
        )
    )

    first = queue.drain()
    second = queue.drain()

    assert first == (make_event(1), make_event(2))
    assert second == ()
    assert queue.empty is True
    assert queue.size == 0


def test_get_raises_for_empty_queue() -> None:
    queue = ReturnEventQueue()

    with pytest.raises(IndexError):
        queue.get()


def test_put_rejects_non_return_event() -> None:
    queue = ReturnEventQueue()

    with pytest.raises(TypeError):
        queue.put("not an event")  # type: ignore[arg-type]

    assert queue.empty is True


def test_put_many_can_receive_an_iterable() -> None:
    queue = ReturnEventQueue()

    queue.put_many(
        make_event(event_id)
        for event_id in (4, 5, 6)
    )

    assert queue.drain() == (
        make_event(4),
        make_event(5),
        make_event(6),
    )
