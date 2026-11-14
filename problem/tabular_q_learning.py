from __future__ import annotations

from dataclasses import dataclass, field

from problem.problem import ManufacturingProblem
from problem.q_learning import QLearningUpdater
from problem.q_learning_agent import QLearningAgent
from problem.q_table import QTable
from problem.reachable_state_registry import (
    ReachableStateRegistry,
)
from problem.state_action_registry import (
    StateActionVisitRegistry,
)
from problem.tabular_state import TabularState
from problem.tabular_transition import TabularTransition


@dataclass
class TabularQLearner:
    """
    Coordinate tabular action selection and Q-learning updates.

    The learner has no dependency on Plant Simulation, TCP,
    tensors, or neural-network components.
    """

    agent: QLearningAgent
    updater: QLearningUpdater
    problem: ManufacturingProblem

    feasible: bool = True

    state_registry: ReachableStateRegistry = field(
        default_factory=ReachableStateRegistry
    )

    state_action_registry: StateActionVisitRegistry = field(
        default_factory=StateActionVisitRegistry
    )

    last_state: TabularState | None = field(
        default=None,
        init=False,
        repr=False,
    )

    last_q_state: TabularState | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, bool):
            raise TypeError(
                "feasible must be a boolean."
            )


    def select_action(
        self,
        state: TabularState,
    ) -> int:
        """
        Select an action and record the resulting decision.

        A state and its selected action are registered after
        successful action selection.
        """

        action = self.agent.select_action(state)

        self.state_registry.observe(state)
        self.state_action_registry.observe(
            state,
            action,
        )

        self.last_state = state

        if any(
            (state, action) in self.agent.q_table.snapshot()
            for action in self.problem.actions
        ):
            self.last_q_state = state

        return action

    def learn(
        self,
        transition: TabularTransition,
    ) -> float:
        """
        Apply one Q-learning update to a transition.
        """

        actions = self._learning_actions(
            transition.next_state
        )

        return self.updater.update(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            terminated=transition.terminated,
            actions=actions,
        )

    def snapshot(self) -> dict[str, object]:
        """
        Return the persistent learning state.
        """
        return {
            "q_table": self.agent.q_table.snapshot(),
            "agent_state": self.agent.snapshot(),
        }

    def restore(
        self,
        state: dict[str, object],
    ) -> None:
        """
        Restore the persistent learning state.
        """
        q_table = state["q_table"]
        agent_state = state["agent_state"]

        if not isinstance(q_table, dict):
            raise TypeError(
                "q_table snapshot must be a dictionary."
            )

        if not isinstance(agent_state, dict):
            raise TypeError(
                "agent_state must be a dictionary."
            )

        self.agent.q_table.restore(q_table)
        self.agent.restore(agent_state)

    def _learning_actions(
        self,
        state: TabularState,
    ) -> tuple[int, ...]:
        """Return the actions considered by the Q-learning target."""

        if self.feasible:
            return self.problem.feasible_actions(state)

        return self.problem.actions