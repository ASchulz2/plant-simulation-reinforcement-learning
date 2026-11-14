from __future__ import annotations

from dataclasses import dataclass
import random

from problem.problem import ManufacturingProblem
from problem.q_table import QTable
from problem.tabular_state import TabularState


@dataclass(frozen=True)
class QLearningAgentSettings:
    """
    Configuration of epsilon-greedy action selection.

    Parameters
    ----------
    epsilon_start:
        Initial exploration probability.

    epsilon_min:
        Minimum exploration probability.

    epsilon_decay:
        Multiplicative decay factor applied after each action selection.

    feasible:
        Whether action selection is restricted to actions that are
        feasible in the current manufacturing state.
    """

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.998
    feasible: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon_start must be in [0, 1]."
            )

        if not 0.0 <= self.epsilon_min <= 1.0:
            raise ValueError(
                "epsilon_min must be in [0, 1]."
            )

        if self.epsilon_min > self.epsilon_start:
            raise ValueError(
                "epsilon_min must not exceed epsilon_start."
            )

        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be in (0, 1]."
            )

        if not isinstance(self.feasible, bool):
            raise TypeError(
                "feasible must be a boolean."
            )


class QLearningAgent:
    """
    Select feasible actions from a Q-table using epsilon-greedy
    exploration.

    The agent is responsible for action selection and management of
    the exploration schedule. It does not perform Q-value updates;
    learning is handled separately by the Q-learning learner and
    updater.

    Parameters
    ----------
    q_table:
        Tabular representation of the learned action-value function.

    problem:
        Manufacturing problem defining the feasible actions for
        each tabular state.

    settings:
        Configuration of the epsilon-greedy exploration schedule.

    rng:
        Optional random-number generator used for exploratory action
        selection. When omitted, a new ``random.Random`` instance is
        created.
    """

    def __init__(
        self,
        q_table: QTable,
        problem: ManufacturingProblem,
        settings: QLearningAgentSettings,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.q_table = q_table
        self.problem = problem
        self.settings = settings
        self.rng = (
            rng
            if rng is not None
            else random.Random()
        )

        self._selection_count = 0

    @property
    def epsilon(self) -> float:
        """
        Return the current exploration probability.

        The exploration probability is computed from the number of
        previous action selections as

        ``max(epsilon_min, epsilon_start * epsilon_decay**n)``

        where ``n`` is the current action-selection count.

        Returns
        -------
        float
            Current exploration probability. The value is bounded
            below by ``epsilon_min``.
        """

        return max(
            self.settings.epsilon_min,
            self.settings.epsilon_start
            * (
                self.settings.epsilon_decay
                ** self._selection_count
            ),
        )

    def select_action(
        self,
        state: TabularState,
    ) -> int:
        """
        Select a feasible action using the epsilon-greedy policy.

        With probability ``epsilon``, an action is sampled from the
        feasible action set. Otherwise, the feasible action with the
        largest stored Q-value is selected.

        Parameters
        ----------
        state:
            Current discrete manufacturing state.

        Returns
        -------
        int
            Selected feasible action.

        Raises
        ------
        ValueError
            If no feasible action exists for the supplied state.
        """

        candidate_actions = self._candidate_actions(state)

        if not candidate_actions:
            raise ValueError(
                "No candidate actions exist for state."
            )

        if not candidate_actions:
            raise ValueError(
                "No feasible actions exist for state."
            )

        if self.rng.random() < self.epsilon:
            action = self.rng.choice(
                candidate_actions
            )
        else:
            action = self._greedy_action(
                state,
                candidate_actions,
            )

        self._selection_count += 1

        return action

    def _greedy_action(
        self,
        state: TabularState,
        candidate_actions: tuple[int, ...],
    ) -> int:
        """
        Select the feasible action with the highest Q-value.

        Parameters
        ----------
        state:
            Current discrete manufacturing state.

        feasible_actions:
            Actions permitted by the manufacturing problem.

        Returns
        -------
        int
            Feasible action with the maximum stored Q-value.
        """

        return max(
            candidate_actions,
            key=lambda action: self.q_table.get(
                state,
                action,
            ),
        )

    def snapshot(self) -> dict[str, object]:
        """
        Return the mutable agent state required for checkpointing.

        The snapshot contains the number of action selections and the
        state of the agent-specific random-number generator. Together,
        these values allow the exploration schedule and random
        behaviour to be restored when training is resumed.

        Returns
        -------
        dict[str, object]
            Serializable agent state containing ``selection_count``
            and ``rng_state``.
        """

        return {
            "selection_count": self._selection_count,
            "rng_state": self.rng.getstate(),
        }

    def restore(
        self,
        state: dict[str, object],
    ) -> None:
        """
        Restore mutable agent state from a checkpoint.

        Parameters
        ----------
        state:
            Previously stored agent state containing ``selection_count``
            and ``rng_state``.

        Raises
        ------
        KeyError
            If a required state entry is missing.

        TypeError
            If ``selection_count`` does not contain an integer.

        ValueError
            If ``selection_count`` is negative or the stored random
            state is invalid.
        """

        if "selection_count" not in state:
            raise KeyError(
                "Agent state does not contain 'selection_count'."
            )

        if "rng_state" not in state:
            raise KeyError(
                "Agent state does not contain 'rng_state'."
            )

        selection_count = state["selection_count"]

        if not isinstance(selection_count, int):
            raise TypeError(
                "selection_count must be an integer."
            )

        if selection_count < 0:
            raise ValueError(
                "selection_count must be non-negative."
            )

        rng_state = state["rng_state"]

        try:
            self.rng.setstate(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid RNG state."
            ) from exc

        self._selection_count = selection_count

    def _candidate_actions(
        self,
        state: TabularState,
    ) -> tuple[int, ...]:
        """
        Return the actions considered by the epsilon-greedy policy.

        Parameters
        ----------
        state:
            Current discrete manufacturing state.

        Returns
        -------
        tuple[int, ...]
            Feasible actions when feasibility filtering is enabled;
            otherwise all actions defined by the manufacturing problem.
        """
        if self.settings.feasible:
            return self.problem.feasible_actions(state)

        return self.problem.actions
