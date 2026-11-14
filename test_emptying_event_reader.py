from unittest.mock import Mock

from environment.emptying_event_reader import (
    EmptyingEventReader,
    EmptyingEventReaderSettings,
)


def make_plant(values: dict[str, object]) -> Mock:
    plant = Mock()

    def get_value(path: str):
        return values[path]

    plant.get_value.side_effect = get_value

    return plant


def make_values(
    *,
    storage_a_event_id: int = 0,
    storage_a_mode: str = "",
    storage_a_count: int = 0,
    storage_a_waiting_time: float = 0.0,
    storage_b_event_id: int = 0,
    storage_b_mode: str = "",
    storage_b_count: int = 0,
    storage_b_waiting_time: float = 0.0,
) -> dict[str, object]:
    settings = EmptyingEventReaderSettings()

    return {
        settings.storage_a_event_id_path: storage_a_event_id,
        settings.storage_a_mode_path: storage_a_mode,
        settings.storage_a_count_path: storage_a_count,
        settings.storage_a_total_waiting_time_path: (
            storage_a_waiting_time
        ),
        settings.storage_b_event_id_path: storage_b_event_id,
        settings.storage_b_mode_path: storage_b_mode,
        settings.storage_b_count_path: storage_b_count,
        settings.storage_b_total_waiting_time_path: (
            storage_b_waiting_time
        ),
    }


def test_reader_returns_new_storage_a_event():
    plant = make_plant(
        make_values(
            storage_a_event_id=1,
            storage_a_mode="Drain",
            storage_a_count=3,
            storage_a_waiting_time=172.0,
        )
    )

    reader = EmptyingEventReader(plant)

    events = reader.read()

    assert len(events) == 1

    event = events[0]

    assert event.storage == "A"
    assert event.event_id == 1
    assert event.mode == "Drain"
    assert event.count == 3
    assert event.total_waiting_time == 172.0


def test_reader_returns_new_storage_b_event():
    plant = make_plant(
        make_values(
            storage_b_event_id=1,
            storage_b_mode="Return",
            storage_b_count=2,
            storage_b_waiting_time=48.5,
        )
    )

    reader = EmptyingEventReader(plant)

    events = reader.read()

    assert len(events) == 1

    event = events[0]

    assert event.storage == "B"
    assert event.event_id == 1
    assert event.mode == "Return"
    assert event.count == 2
    assert event.total_waiting_time == 48.5


def test_reader_returns_both_new_events():
    plant = make_plant(
        make_values(
            storage_a_event_id=3,
            storage_a_mode="Drain",
            storage_a_count=3,
            storage_a_waiting_time=100.0,
            storage_b_event_id=5,
            storage_b_mode="Return",
            storage_b_count=1,
            storage_b_waiting_time=20.0,
        )
    )

    reader = EmptyingEventReader(plant)

    events = reader.read()

    assert len(events) == 2

    assert events[0].storage == "A"
    assert events[0].event_id == 3

    assert events[1].storage == "B"
    assert events[1].event_id == 5


def test_reader_does_not_emit_same_event_twice():
    plant = make_plant(
        make_values(
            storage_a_event_id=1,
            storage_a_mode="Drain",
            storage_a_count=3,
            storage_a_waiting_time=172.0,
        )
    )

    reader = EmptyingEventReader(plant)

    first_events = reader.read()
    second_events = reader.read()

    assert len(first_events) == 1
    assert len(second_events) == 0


def test_reader_emits_event_when_event_id_increases():
    values = make_values(
        storage_a_event_id=1,
        storage_a_mode="Drain",
        storage_a_count=3,
        storage_a_waiting_time=100.0,
    )

    plant = make_plant(values)
    reader = EmptyingEventReader(plant)

    first_events = reader.read()

    assert len(first_events) == 1
    assert first_events[0].event_id == 1

    values[
        EmptyingEventReaderSettings().storage_a_event_id_path
    ] = 2

    values[
        EmptyingEventReaderSettings().storage_a_mode_path
    ] = "Return"

    values[
        EmptyingEventReaderSettings().storage_a_count_path
    ] = 2

    values[
        EmptyingEventReaderSettings().storage_a_total_waiting_time_path
    ] = 50.0

    second_events = reader.read()

    assert len(second_events) == 1
    assert second_events[0].event_id == 2
    assert second_events[0].mode == "Return"
    assert second_events[0].count == 2
    assert second_events[0].total_waiting_time == 50.0


def test_reset_allows_previous_event_id_to_be_read_again():
    plant = make_plant(
        make_values(
            storage_a_event_id=1,
            storage_a_mode="Drain",
            storage_a_count=3,
            storage_a_waiting_time=172.0,
        )
    )

    reader = EmptyingEventReader(plant)

    assert len(reader.read()) == 1
    assert len(reader.read()) == 0

    reader.reset()

    events = reader.read()

    assert len(events) == 1
    assert events[0].event_id == 1