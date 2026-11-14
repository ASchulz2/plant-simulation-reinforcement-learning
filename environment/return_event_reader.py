from __future__ import annotations

from dataclasses import dataclass

from .return_event import ReturnEvent


@dataclass(frozen=True)
class ReturnEventReaderSettings:
    """Plant Simulation interface for queued ReturnBuffer events."""

    has_event_path: str = (
        ".Models.Model.ReturnBuffer.RL_hasReturnEvent"
    )

    pop_event_path: str = (
        ".Models.Model.ReturnBuffer.RL_popReturnEvent"
    )


class ReturnEventReader:
    """Consume queued ReturnBuffer return events.

    The Plant side stores each completed return event as a serialized
    string with the format:

        event_id|language|waiting_time

    The reader consumes all currently queued events in FIFO order.
    """

    def __init__(
        self,
        plant,
        settings: ReturnEventReaderSettings | None = None,
    ) -> None:
        self.plant = plant
        self.settings = settings or ReturnEventReaderSettings()

    def reset(self) -> None:
        """Reset reader state.

        The event queue belongs to Plant Simulation and is therefore
        not cleared by the reader.
        """
        return None

    def read(self) -> tuple[ReturnEvent, ...]:
        """Consume all currently queued ReturnBuffer events."""

        events: list[ReturnEvent] = []

        while self._has_event():
            serialized_event = self._pop_event()

            event = self._parse_event(
                serialized_event
            )

            events.append(event)

        return tuple(events)

    def _has_event(self) -> bool:
        """Return whether a ReturnBuffer event is available."""

        return bool(
            self.plant.get_value(
                self.settings.has_event_path
            )
        )

    def _pop_event(self) -> str:
        """Remove and return the oldest ReturnBuffer event."""

        return str(
            self.plant.get_value(
                self.settings.pop_event_path
            )
        )

    @staticmethod
    def _parse_event(
        serialized_event: str,
    ) -> ReturnEvent:
        """Parse one serialized ReturnBuffer event."""

        parts = serialized_event.split("|")

        if len(parts) != 3:
            raise ValueError(
                "Invalid ReturnBuffer event: "
                f"{serialized_event!r}. Expected "
                "'event_id|language|waiting_time'."
            )

        event_id_text, language, waiting_time_text = parts

        try:
            event_id = int(event_id_text)
        except ValueError as exc:
            raise ValueError(
                "Invalid ReturnBuffer event ID: "
                f"{event_id_text!r}"
            ) from exc

        try:
            waiting_time = ReturnEventReader._parse_waiting_time(
                waiting_time_text
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid ReturnBuffer waiting time: "
                f"{waiting_time_text!r}"
            ) from exc

        return ReturnEvent(
            event_id=event_id,
            language=language,
            waiting_time=waiting_time,
        )

    @staticmethod
    def _parse_waiting_time(value: str) -> float:
        value = value.strip()

        # Plant Simulation Time is fundamentally represented in seconds.
        # Accept a plain numeric representation as well.
        try:
            return float(value)
        except ValueError:
            pass

        parts = value.split(":")

        if len(parts) == 2:
            # mm:ss.ss
            minutes_text, seconds_text = parts
            hours = 0.0
        elif len(parts) == 3:
            # hh:mm:ss.ss
            hours_text, minutes_text, seconds_text = parts

            try:
                hours = float(hours_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Plant Simulation time: {value!r}"
                ) from exc
        else:
            raise ValueError(
                f"Invalid Plant Simulation time: {value!r}"
            )

        try:
            minutes = float(minutes_text)
            seconds = float(seconds_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid Plant Simulation time: {value!r}"
            ) from exc

        if hours < 0 or minutes < 0 or seconds < 0:
            raise ValueError(
                f"Invalid Plant Simulation time: {value!r}"
            )

        return float(
            hours * 3600.0
            + minutes * 60.0
            + seconds
        )