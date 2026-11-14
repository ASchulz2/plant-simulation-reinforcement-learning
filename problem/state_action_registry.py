from __future__ import annotations

from dataclasses import dataclass, field

from problem.tabular_state import TabularState


@dataclass
class StateActionVisitRegistry:
    """
    Track empirical visitation counts for state-action pairs.

    This class records experience only. It does not select actions,
    update Q-values, or perform frontier learning.
    """

    _visits: dict[tuple[TabularState, int], int] = field(
        default_factory=dict
    )

    def observe(
        self,
        state: TabularState,
        action: int,
    ) -> None:
        """Record one observation of a state-action pair."""

        key = (state, action)

        self._visits[key] = (
            self._visits.get(key, 0) + 1
        )

    def contains(
        self,
        state: TabularState,
        action: int,
    ) -> bool:
        """Return whether the state-action pair has been observed."""

        return (state, action) in self._visits

    def visits(
        self,
        state: TabularState,
        action: int,
    ) -> int:
        """Return the visitation count of a state-action pair."""

        return self._visits.get(
            (state, action),
            0,
        )

    def count(self) -> int:
        """Return the number of distinct observed state-action pairs."""

        return len(self._visits)

    def state_actions(
        self,
    ) -> tuple[tuple[TabularState, int], ...]:
        """
        Return observed state-action pairs in first-observation order.
        """

        return tuple(self._visits)