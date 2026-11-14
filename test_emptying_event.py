import pytest

from environment.emptying_event import EmptyingEvent


def test_emptying_event_stores_values():
    event = EmptyingEvent(
        storage="A",
        event_id=7,
        mode="Drain",
        count=3,
        total_waiting_time=172.5,
    )

    assert event.storage == "A"
    assert event.event_id == 7
    assert event.mode == "Drain"
    assert event.count == 3
    assert event.total_waiting_time == 172.5


def test_drain_event_is_classified_correctly():
    event = EmptyingEvent(
        storage="A",
        event_id=1,
        mode="Drain",
        count=3,
        total_waiting_time=10.0,
    )

    assert event.is_drain is True
    assert event.is_return is False


def test_return_event_is_classified_correctly():
    event = EmptyingEvent(
        storage="B",
        event_id=2,
        mode="Return",
        count=2,
        total_waiting_time=25.0,
    )

    assert event.is_drain is False
    assert event.is_return is True


def test_emptying_event_is_immutable():
    event = EmptyingEvent(
        storage="A",
        event_id=1,
        mode="Drain",
        count=3,
        total_waiting_time=10.0,
    )

    with pytest.raises(AttributeError):
        event.mode = "Return"