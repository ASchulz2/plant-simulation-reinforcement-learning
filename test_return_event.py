import pytest

from environment.return_event import ReturnEvent


def test_return_event_stores_values() -> None:
    event = ReturnEvent(
        event_id=7,
        language="FR",
        waiting_time=41.5,
    )

    assert event.event_id == 7
    assert event.language == "FR"
    assert event.waiting_time == 41.5


def test_return_event_rejects_non_positive_id() -> None:
    with pytest.raises(ValueError, match="event_id"):
        ReturnEvent(
            event_id=0,
            language="FR",
            waiting_time=10.0,
        )


def test_return_event_rejects_empty_language() -> None:
    with pytest.raises(ValueError, match="language"):
        ReturnEvent(
            event_id=1,
            language="",
            waiting_time=10.0,
        )


def test_return_event_rejects_negative_waiting_time() -> None:
    with pytest.raises(ValueError, match="waiting_time"):
        ReturnEvent(
            event_id=1,
            language="FR",
            waiting_time=-1.0,
        )
