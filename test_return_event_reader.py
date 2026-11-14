from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from environment.return_event_reader import (
    ReturnEventReader,
    ReturnEventReaderSettings,
)
from environment.return_event_reader import ReturnEventReader

@dataclass
class FakePlant:
    events: list[str] = field(default_factory=list)

    def get_value(self, path: str) -> object:
        if path == "has_event":
            return bool(self.events)

        if path == "pop_event":
            if not self.events:
                raise RuntimeError(
                    "Attempted to pop from an empty event queue."
                )

            return self.events.pop(0)

        raise KeyError(path)


def make_reader(
    plant: FakePlant,
) -> ReturnEventReader:
    return ReturnEventReader(
        plant=plant,
        settings=ReturnEventReaderSettings(
            has_event_path="has_event",
            pop_event_path="pop_event",
        ),
    )


def test_reader_returns_all_queued_events() -> None:
    plant = FakePlant(
        events=[
            "1|FR|41.6623",
            "2|FR|39.0172",
            "3|FR|36.0000",
        ]
    )

    reader = make_reader(plant)

    events = reader.read()

    assert [event.event_id for event in events] == [1, 2, 3]
    assert [event.language for event in events] == [
        "FR",
        "FR",
        "FR",
    ]
    assert [event.waiting_time for event in events] == [
        41.6623,
        39.0172,
        36.0,
    ]

    assert plant.events == []


def test_reader_returns_empty_when_queue_is_empty() -> None:
    plant = FakePlant()

    reader = make_reader(plant)

    assert reader.read() == ()


def test_reader_consumes_events_in_fifo_order() -> None:
    plant = FakePlant(
        events=[
            "5|FR|36.0000",
            "6|RUS|36.0000",
            "7|GER|37.6691",
        ]
    )

    reader = make_reader(plant)

    events = reader.read()

    assert [event.event_id for event in events] == [
        5,
        6,
        7,
    ]

    assert [event.language for event in events] == [
        "FR",
        "RUS",
        "GER",
    ]


def test_reader_can_read_events_added_after_previous_read() -> None:
    plant = FakePlant(
        events=[
            "1|FR|36.0000",
        ]
    )

    reader = make_reader(plant)

    first_events = reader.read()

    assert len(first_events) == 1
    assert first_events[0].event_id == 1

    plant.events.extend(
        [
            "2|FR|38.5594",
            "3|RUS|43.5103",
        ]
    )

    second_events = reader.read()

    assert [event.event_id for event in second_events] == [
        2,
        3,
    ]


def test_reader_reset_does_not_clear_plant_queue() -> None:
    plant = FakePlant(
        events=[
            "1|FR|36.0000",
        ]
    )

    reader = make_reader(plant)

    reader.reset()

    events = reader.read()

    assert len(events) == 1
    assert events[0].event_id == 1


def test_reader_rejects_invalid_event_format() -> None:
    plant = FakePlant(
        events=[
            "1|FR",
        ]
    )

    reader = make_reader(plant)

    with pytest.raises(
        ValueError,
        match="Invalid ReturnBuffer event",
    ):
        reader.read()


def test_reader_rejects_invalid_event_id() -> None:
    plant = FakePlant(
        events=[
            "abc|FR|36.0000",
        ]
    )

    reader = make_reader(plant)

    with pytest.raises(
        ValueError,
        match="Invalid ReturnBuffer event ID",
    ):
        reader.read()


def test_reader_rejects_invalid_waiting_time() -> None:
    plant = FakePlant(
        events=[
            "1|FR|invalid",
        ]
    )

    reader = make_reader(plant)

    with pytest.raises(
        ValueError,
        match="Invalid ReturnBuffer waiting time",
    ):
        reader.read()


def test_default_settings() -> None:
    settings = ReturnEventReaderSettings()

    assert settings.has_event_path == (
        ".Models.Model.ReturnBuffer.RL_hasReturnEvent"
    )

    assert settings.pop_event_path == (
        ".Models.Model.ReturnBuffer.RL_popReturnEvent"
    )


def test_settings_expose_method_interface() -> None:
    settings = ReturnEventReaderSettings()

    assert hasattr(settings, "has_event_path")
    assert hasattr(settings, "pop_event_path")

    assert not hasattr(settings, "queue_path")
    assert not hasattr(settings, "event_id_path")
    assert not hasattr(settings, "waiting_time_path")
    assert not hasattr(settings, "language_path")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.5", 10.5),
        ("2:27.5962", 147.5962),
        ("00:02:27.5962", 147.5962),
        ("01:02:03.25", 3723.25),
        ("00:00:05.25", 5.25),
    ],
)
def test_parse_waiting_time(value, expected):
    result = ReturnEventReader._parse_waiting_time(value)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "abc",
        "1:2:3:4",
        "1:abc",
        "abc:2.5",
        "1:2:abc",
    ],
)
def test_parse_waiting_time_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        ReturnEventReader._parse_waiting_time(value)

def test_parse_event_converts_plant_simulation_time():
    event = ReturnEventReader._parse_event(
        "42|GER|00:02:27.5962"
    )

    assert event.event_id == 42
    assert event.language == "GER"
    assert event.waiting_time == pytest.approx(147.5962)