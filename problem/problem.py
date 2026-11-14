from __future__ import annotations

from dataclasses import dataclass

from problem.observation import Observation
from problem.tabular_state import TabularState


@dataclass(frozen=True)
class ManufacturingProblem:
    """
    Definition of the board-game manufacturing decision problem.

    This class contains problem semantics only. It does not know about
    Plant Simulation, TCP, PPO, or the agent implementation.
    """

    target_throughput: int = 1000

    admission_buffer_capacity: int = 10

    # Maximum number of MUs that can be present in one
    # intermediate storage at a decision point.
    storage_capacity: int = 10

    # Discrete routing actions.
    # 0 = Storage A
    # 1 = Storage B
    # 2 = Return
    actions: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if not isinstance(
            self.storage_capacity,
            int,
        ):
            raise TypeError(
                "storage_capacity must be an integer."
            )

        if self.storage_capacity <= 0:
            raise ValueError(
                "storage_capacity must be positive."
            )
        if not isinstance(
            self.admission_buffer_capacity,
            int,
        ):
            raise TypeError(
                "admission_buffer_capacity must be an integer."
            )

        if self.admission_buffer_capacity <= 0:
            raise ValueError(
                "admission_buffer_capacity must be positive."
            )

    def validate_action(
        self,
        action: int,
    ) -> None:
        """
        Validate that an action belongs to the problem's action space.
        """

        if not isinstance(action, int):
            raise TypeError(
                "Action must be an integer; "
                f"got {type(action).__name__}."
            )

        if action not in self.actions:
            raise ValueError(
                f"Invalid action {action}. "
                f"Expected one of {self.actions}."
            )

    def feasible_actions(
        self,
        state: TabularState,
    ) -> tuple[int, ...]:
        """
        Return the actions that are semantically feasible in a tabular state.

        A storage action is feasible only when:
            1. the storage is not full, and
            2. the storage is empty or already contains the
               current language.

        Return is always feasible when a current MU exists.
        """

        current_language = state.current_language

        if current_language == "":
            return ()

        feasible: list[int] = []

        if (
            state.storage_a_count < self.storage_capacity
            and (
                state.storage_a_language == ""
                or (
                    state.storage_a_language
                    == current_language
                )
            )
        ):
            feasible.append(0)

        if (
            state.storage_b_count < self.storage_capacity
            and (
                state.storage_b_language == ""
                or (
                    state.storage_b_language
                    == current_language
                )
            )
        ):
            feasible.append(1)

        feasible.append(2)

        return tuple(feasible)

    def is_successful(
        self,
        observation: Observation,
    ) -> bool:
        """
        Return whether the manufacturing objective has been achieved.
        """

        return (
            observation.throughput
            >= self.target_throughput
        )
