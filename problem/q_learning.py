from __future__ import annotations

from dataclasses import dataclass

from problem.q_table import QTable
from problem.tabular_state import TabularState


@dataclass(frozen=True)
class QLearningSettings:
    """Configuration of the tabular Q-learning update."""

    learning_rate: float = 0.1
    discount_factor: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(
                "learning_rate must be in (0, 1]."
            )

        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be in [0, 1]."
            )


class QLearningUpdater:
    """
    Perform Bellman updates on a QTable.

    This class contains learning logic only. It does not select
    actions and has no dependency on the environment.
    """

    def __init__(
        self,
        q_table: QTable,
        settings: QLearningSettings,
    ) -> None:
        self.q_table = q_table
        self.settings = settings

    def update(
        self,
        state: TabularState,
        action: int,
        reward: float,
        next_state: TabularState,
        *,
        terminated: bool,
        actions: tuple[int, ...],
    ) -> float:
        """
        Update Q(state, action) and return the new Q-value.
        """

        current_value = self.q_table.get(
            state,
            action,
        )

        if terminated:
            target = reward

        else:
            if not actions:
                raise ValueError(
                    "At least one action is required "
                    "for a non-terminal update."
                )

            next_value = max(
                self.q_table.get(
                    next_state,
                    next_action,
                )
                for next_action in actions
            )

            target = (
                reward
                + self.settings.discount_factor
                * next_value
            )

        updated_value = (
            current_value
            + self.settings.learning_rate
            * (target - current_value)
        )

        self.q_table.set(
            state,
            action,
            updated_value,
        )

        return updated_value