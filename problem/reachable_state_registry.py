from __future__ import annotations

from dataclasses import dataclass, field

from problem.tabular_state import TabularState


@dataclass
class ReachableStateRegistry:
    """
    Registry of tabular states observed during interaction.

    The registry is deliberately independent of Q-learning,
    frontier selection, and the simulation environment.
    """

    _visits: dict[TabularState, int] = field(
        default_factory=dict
    )

    def observe(
        self,
        state: TabularState,
    ) -> None:
        """Record one visit to a state."""

        self._visits[state] = (
            self._visits.get(state, 0) + 1
        )

    def contains(
        self,
        state: TabularState,
    ) -> bool:
        """Return whether the state has been observed."""

        return state in self._visits

    def count(self) -> int:
        """Return the number of distinct observed states."""

        return len(self._visits)

    def visits(
        self,
        state: TabularState,
    ) -> int:
        """Return the number of times a state was observed."""

        return self._visits.get(
            state,
            0,
        )

    def states(self) -> tuple[TabularState, ...]:
        """
        Return all observed states.

        States are returned in insertion order.
        """

        return tuple(self._visits)