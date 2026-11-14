from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpace:
    """
    Finite discrete action space for the manufacturing problem.
    """

    actions: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError(
                "Action space must contain at least one action."
            )

        if len(set(self.actions)) != len(self.actions):
            raise ValueError(
                "Action space must not contain duplicate actions."
            )

        if any(
            not isinstance(action, int)
            for action in self.actions
        ):
            raise TypeError(
                "All actions must be integers."
            )

    def contains(
        self,
        action: int,
    ) -> bool:
        return action in self.actions

    def validate(
        self,
        action: int,
    ) -> None:
        if not isinstance(action, int):
            raise TypeError(
                f"Action must be int, "
                f"got {type(action).__name__}."
            )

        if action not in self.actions:
            raise ValueError(
                f"Invalid action {action}. "
                f"Valid actions are {self.actions}."
            )