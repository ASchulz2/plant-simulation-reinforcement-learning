from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from environment.emptying_event import EmptyingEvent
from environment.return_event import ReturnEvent
from problem.observation import Observation


class RewardFunction(ABC):
    """Abstract reward function for semantic transitions."""

    @abstractmethod
    def calculate(
        self,
        previous_state: Observation,
        next_state: Observation,
        *,
        action: int,
        emptying_events: tuple[EmptyingEvent, ...] = (),
        return_events: tuple[ReturnEvent, ...] = (),
        terminated: bool = False,
        succeeded: bool = False,
        action_feasible: bool | None = None,
    ) -> float:
        """
        Calculate the scalar reward for one transition.

        Parameters
        ----------
        previous_state:
            Observation before the action.

        next_state:
            Observation after the action.

        action:
            Action taken during the transition.

        emptying_events:
            Newly observed storage emptying events produced during
            this transition.

        return_events:
            Newly observed individual MU return events produced
            during this transition.

        terminated:
            Whether the transition ended the episode.

        succeeded:
            Whether the terminal transition represents success.

        action_feasible:
            Whether the selected action was a feasible candidate in
            the previous state. ``None`` disables the feasibility
            reward component for this transition.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class QLearningRewardSettings:
    """Configuration of the tabular Q-learning reward."""

    throughput_weight: float = 5.0
    storage_reward: float = 1.0
    time_penalty: float = 0.01
    return_waiting_time_penalty: float = 1.0
    return_waiting_time_reference: float = 36.0
    success_bonus: float = 10.0
    failure_penalty: float = 10.0
    premature_emptying_penalty: float = 10.0
    return_action_penalty: float = 1.0
    feasibility_reward: float = 1.0

    def __post_init__(self) -> None:
        if self.throughput_weight < 0.0:
            raise ValueError(
                "throughput_weight must be non-negative."
            )

        if self.storage_reward < 0.0:
            raise ValueError(
                "storage_reward must be non-negative."
            )

        if self.time_penalty < 0.0:
            raise ValueError(
                "time_penalty must be non-negative."
            )

        if self.return_waiting_time_penalty < 0.0:
            raise ValueError(
                "return_waiting_time_penalty must be non-negative."
            )

        if self.return_waiting_time_reference <= 0.0:
            raise ValueError(
                "return_waiting_time_reference must be positive."
            )

        if self.success_bonus < 0.0:
            raise ValueError(
                "success_bonus must be non-negative."
            )

        if self.failure_penalty < 0.0:
            raise ValueError(
                "failure_penalty must be non-negative."
            )

        if self.premature_emptying_penalty < 0.0:
            raise ValueError(
                "premature_emptying_penalty must be non-negative."
            )

        if self.return_action_penalty < 0.0:
            raise ValueError(
                "return_action_penalty must be non-negative."
            )

        if self.feasibility_reward < 0.0:
            raise ValueError(
                "feasibility_reward must be non-negative."
            )


class QLearningReward(RewardFunction):
    """
    Transition reward for the tabular Q-learning baseline.

    Reward components:

        throughput progress
        + positive storage-occupancy progress
        - elapsed simulation-time cost
        - normalized waiting-time cost for returned MUs
        + terminal success bonus
        - terminal failure penalty

    Storage emptying events receive a penalty when a returnLoop
    empties the storage before reaching its capacity. The penalty
    decreases linearly with the number of MUs in the emptied batch.
    Return emptying events do not receive this penalty.

    Return waiting times are normalized by the configured reference
    waiting time before applying the return-waiting coefficient.
    """

    def __init__(
        self,
        settings: QLearningRewardSettings | None = None,
        *,
        storage_capacity: int = 10,
    ) -> None:
        self.settings = (
            settings
            if settings is not None
            else QLearningRewardSettings()
        )
        self.storage_capacity = storage_capacity

        if storage_capacity <= 0:
            raise ValueError(
                "storage_capacity must be positive."
        )

    def _return_action_penalty(
        self,
        action: int,
        ) -> float:
        if action == 2:
            return self.settings.return_action_penalty

        return 0.0

    def calculate(
        self,
        previous_state: Observation,
        next_state: Observation,
        *,
        action: int,
        emptying_events: tuple[EmptyingEvent, ...] = (),
        return_events: tuple[ReturnEvent, ...] = (),
        terminated: bool = False,
        succeeded: bool = False,
        action_feasible: bool | None = None,
    ) -> float:
        """Calculate the reward for one Q-learning transition."""

        # The action is intentionally not interpreted here.
        # ManufacturingProblem owns action-space validation
        # and action semantics.

        # --------------------------------------------------------------
        # Throughput progress.
        # --------------------------------------------------------------

        delta_throughput = (
            next_state.throughput
            - previous_state.throughput
        )

        if delta_throughput < 0:
            raise ValueError(
                "Throughput decreased between "
                "two consecutive states."
            )

        reward = (
            self.settings.throughput_weight
            * delta_throughput
        )

        # --------------------------------------------------------------
        # Storage occupancy progress.
        # --------------------------------------------------------------

        delta_storage_a = (
            next_state.storage_a_count
            - previous_state.storage_a_count
        )

        delta_storage_b = (
            next_state.storage_b_count
            - previous_state.storage_b_count
        )

        positive_storage_increment = (
            max(0, delta_storage_a)
            + max(0, delta_storage_b)
        )

        reward += (
            self.settings.storage_reward
            * positive_storage_increment
        )

        # --------------------------------------------------------------
        # Elapsed simulation time.
        # --------------------------------------------------------------

        delta_time = (
            next_state.simulation_time
            - previous_state.simulation_time
        )

        if delta_time < 0.0:
            raise ValueError(
                "Simulation time decreased between "
                "two consecutive states."
            )

        reward -= (
            self.settings.time_penalty
            * delta_time
        )

        # --------------------------------------------------------------
        # Returned-MU waiting-time penalty.
        # --------------------------------------------------------------

        total_normalized_return_waiting_time = sum(
            event.waiting_time
            / self.settings.return_waiting_time_reference
            for event in return_events
        )

        reward -= (
            self.settings.return_waiting_time_penalty
            * total_normalized_return_waiting_time
        )

        # --------------------------------------------------------------
        # Premature Emptying Penalty
        # --------------------------------------------------------------
        total_premature_emptying_penalty = sum(
            self._premature_emptying_penalty(
                event,
                storage_capacity=self.storage_capacity,
            )
            for event in emptying_events
        )

        reward -= total_premature_emptying_penalty

        reward -= self._return_action_penalty(action)


        # --------------------------------------------------------------
        # Feasibility Reward
        # --------------------------------------------------------------
        if action_feasible is not None:
            if not isinstance(action_feasible, bool):
                raise TypeError(
                    "action_feasible must be a boolean or None."
                )

            reward += self._feasibility_reward(
                action_feasible
            )

        # --------------------------------------------------------------
        # Terminal reward.
        # --------------------------------------------------------------

        if terminated:
            if succeeded:
                reward += self.settings.success_bonus
            else:
                reward -= self.settings.failure_penalty

        return float(reward)

    def _premature_emptying_penalty(
        self,
        event: EmptyingEvent,
        *,
        storage_capacity: int,
    ) -> float:
        """Calculate the penalty for one premature drain event."""

        if not event.is_drain:
            return 0.0

        if event.count < 0:
            raise ValueError(
                "Emptying event count must be non-negative."
            )

        if event.count > storage_capacity:
            raise ValueError(
                "Emptying event count must not exceed "
                "storage capacity."
            )

        penalty_fraction = (
            1.0 - event.count / storage_capacity
        )

        return (
            self.settings.premature_emptying_penalty
            * penalty_fraction
        )

    def _feasibility_reward(
        self,
        action_feasible: bool,
    ) -> float:
        """Return the reward associated with action feasibility."""

        if action_feasible:
            return self.settings.feasibility_reward

        return -self.settings.feasibility_reward