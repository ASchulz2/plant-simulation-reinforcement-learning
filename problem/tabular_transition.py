from __future__ import annotations

from dataclasses import dataclass

from problem.tabular_state import TabularState


@dataclass(frozen=True)
class TabularTransition:
    """
    One transition for tabular reinforcement learning.
    """

    state: TabularState
    action: int
    reward: float
    next_state: TabularState
    terminated: bool