from __future__ import annotations

from dataclasses import dataclass

from environment.emptying_event import EmptyingEvent


@dataclass(frozen=True)
class EmptyingEventReaderSettings:
    """COM paths for reward-relevant storage event attributes."""

    storage_a_event_id_path: str = (
        ".Models.Model.StorageA.RL_lastEmptyingID"
    )

    storage_a_mode_path: str = (
        ".Models.Model.StorageA.RL_lastEmptyingMode"
    )

    storage_a_count_path: str = (
        ".Models.Model.StorageA.RL_lastEmptyingCount"
    )

    storage_a_total_waiting_time_path: str = (
        ".Models.Model.StorageA.RL_lastTotalWaitingTime"
    )

    storage_b_event_id_path: str = (
        ".Models.Model.StorageB.RL_lastEmptyingID"
    )

    storage_b_mode_path: str = (
        ".Models.Model.StorageB.RL_lastEmptyingMode"
    )

    storage_b_count_path: str = (
        ".Models.Model.StorageB.RL_lastEmptyingCount"
    )

    storage_b_total_waiting_time_path: str = (
        ".Models.Model.StorageB.RL_lastTotalWaitingTime"
    )


class EmptyingEventReader:
    """
    Read reward-relevant storage emptying events from Plant Simulation.

    Plant Simulation publishes the latest event for each storage.
    This reader converts those attributes into EmptyingEvent objects
    and suppresses events that have already been consumed.
    """

    def __init__(
        self,
        plant,
        settings: EmptyingEventReaderSettings | None = None,
    ) -> None:
        self.plant = plant

        self.settings = (
            settings
            if settings is not None
            else EmptyingEventReaderSettings()
        )

        self._last_storage_a_event_id = 0
        self._last_storage_b_event_id = 0

    def read(self) -> tuple[EmptyingEvent, ...]:
        """
        Read all newly published storage emptying events.

        Returns:
            Tuple containing zero, one, or two new events.
        """

        events: list[EmptyingEvent] = []

        storage_a = self._read_storage(
            storage="A",
            event_id_path=(
                self.settings.storage_a_event_id_path
            ),
            mode_path=(
                self.settings.storage_a_mode_path
            ),
            count_path=(
                self.settings.storage_a_count_path
            ),
            total_waiting_time_path=(
                self.settings.storage_a_total_waiting_time_path
            ),
            last_event_id=(
                self._last_storage_a_event_id
            ),
        )

        if storage_a is not None:
            events.append(storage_a)
            self._last_storage_a_event_id = (
                storage_a.event_id
            )

        storage_b = self._read_storage(
            storage="B",
            event_id_path=(
                self.settings.storage_b_event_id_path
            ),
            mode_path=(
                self.settings.storage_b_mode_path
            ),
            count_path=(
                self.settings.storage_b_count_path
            ),
            total_waiting_time_path=(
                self.settings.storage_b_total_waiting_time_path
            ),
            last_event_id=(
                self._last_storage_b_event_id
            ),
        )

        if storage_b is not None:
            events.append(storage_b)
            self._last_storage_b_event_id = (
                storage_b.event_id
            )

        return tuple(events)

    def reset(self) -> None:
        """Reset the reader's consumed-event state."""
        self._last_storage_a_event_id = 0
        self._last_storage_b_event_id = 0

    def _read_storage(
        self,
        *,
        storage: str,
        event_id_path: str,
        mode_path: str,
        count_path: str,
        total_waiting_time_path: str,
        last_event_id: int,
    ) -> EmptyingEvent | None:
        """Read one storage and return a newly published event."""

        event_id = int(
            self.plant.get_value(event_id_path)
        )

        if event_id <= last_event_id:
            return None

        mode = str(
            self.plant.get_value(mode_path)
        )

        count = int(
            self.plant.get_value(count_path)
        )

        total_waiting_time = float(
            self.plant.get_value(
                total_waiting_time_path
            )
        )

        return EmptyingEvent(
            storage=storage,
            event_id=event_id,
            mode=mode,
            count=count,
            total_waiting_time=total_waiting_time,
        )