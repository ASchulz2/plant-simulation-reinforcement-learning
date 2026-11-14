from __future__ import annotations

from itertools import product

from problem.tabular_state import TabularState


class TabularStateSpace:
    """
    Enumerate the theoretical discrete state space of the tabular
    routing representation.

    The state contains:
        - current MU language,
        - AdmissionBuffer language counts,
        - Storage A language and occupancy,
        - Storage B language and occupancy.

    AdmissionBuffer order is deliberately discarded at this level.

    A storage is represented semantically:
        - occupancy 0 -> empty language "",
        - occupancy > 0 -> one of the three languages.
    """

    LANGUAGES = (
        "GER",
        "FR",
        "RUS",
    )

    EMPTY_LANGUAGE = ""

    def __init__(
        self,
        max_buffer_capacity: int,
        storage_capacity: int,
    ) -> None:
        if not isinstance(
            max_buffer_capacity,
            int,
        ):
            raise TypeError(
                "max_buffer_capacity must be an integer."
            )

        if max_buffer_capacity < 0:
            raise ValueError(
                "max_buffer_capacity must be non-negative."
            )

        if not isinstance(
            storage_capacity,
            int,
        ):
            raise TypeError(
                "storage_capacity must be an integer."
            )

        if storage_capacity < 0:
            raise ValueError(
                "storage_capacity must be non-negative."
            )

        self.max_buffer_capacity = (
            max_buffer_capacity
        )

        self.storage_capacity = (
            storage_capacity
        )

    def states(self) -> tuple[TabularState, ...]:
        """
        Enumerate all semantically consistent tabular states.
        """

        states: list[TabularState] = []

        for (
            current_language,
            ger_count,
            fr_count,
            rus_count,
        ) in self._buffer_states():

            for (
                storage_a_language,
                storage_a_count,
            ) in self._storage_states():

                for (
                    storage_b_language,
                    storage_b_count,
                ) in self._storage_states():

                    states.append(
                        TabularState(
                            current_language=current_language,
                            admission_buffer_ger_count=ger_count,
                            admission_buffer_fr_count=fr_count,
                            admission_buffer_rus_count=rus_count,
                            storage_a_language=(
                                storage_a_language
                            ),
                            storage_a_count=(
                                storage_a_count
                            ),
                            storage_b_language=(
                                storage_b_language
                            ),
                            storage_b_count=(
                                storage_b_count
                            ),
                        )
                    )

        return tuple(states)

    def count(self) -> int:
        """Return the number of states in the theoretical state space."""

        return len(self.states())

    def _storage_states(
        self,
    ) -> tuple[tuple[str, int], ...]:
        """
        Enumerate semantically consistent states for one storage.
        """

        states: list[tuple[str, int]] = [
            (self.EMPTY_LANGUAGE, 0)
        ]

        for count in range(
            1,
            self.storage_capacity + 1,
        ):
            for language in self.LANGUAGES:
                states.append(
                    (
                        language,
                        count,
                    )
                )

        return tuple(states)

    def _buffer_states(
        self,
    ) -> tuple[tuple[str, int, int, int], ...]:
        states: list[
            tuple[str, int, int, int]
        ] = []

        for total in range(
            self.max_buffer_capacity + 1
        ):
            for ger_count in range(total + 1):
                for fr_count in range(
                    total - ger_count + 1
                ):
                    rus_count = (
                        total
                        - ger_count
                        - fr_count
                    )

                    if total == 0:
                        states.append(
                            ("", 0, 0, 0)
                        )
                        continue

                    if ger_count > 0:
                        states.append(
                            (
                                "GER",
                                ger_count,
                                fr_count,
                                rus_count,
                            )
                        )

                    if fr_count > 0:
                        states.append(
                            (
                                "FR",
                                ger_count,
                                fr_count,
                                rus_count,
                            )
                        )

                    if rus_count > 0:
                        states.append(
                            (
                                "RUS",
                                ger_count,
                                fr_count,
                                rus_count,
                            )
                        )

        return tuple(states)
