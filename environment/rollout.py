from dataclasses import dataclass, field

import torch
from torch import Tensor

from environment.transition import Transition

@dataclass
class RolloutTensors:
    """Tensor representation of a collected rollout."""

    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor

    old_log_probabilities: Tensor
    values: Tensor

    terminated: Tensor
    elapsed_times: Tensor


@dataclass
class Rollout:
    """Bounded collection of transitions for PPO."""

    horizon: int

    transitions: list[Transition] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(
                "Rollout horizon must be positive."
            )

    def add(
        self,
        transition: Transition,
    ) -> None:
        """Add a transition to the rollout."""

        if self.is_full:
            raise RuntimeError(
                "Rollout is already full."
            )

        self.transitions.append(transition)

    @property
    def length(self) -> int:
        return len(self.transitions)

    @property
    def is_empty(self) -> bool:
        return self.length == 0

    @property
    def is_full(self) -> bool:
        return self.length >= self.horizon

    def clear(self) -> None:
        self.transitions.clear()

    def observations(self) -> Tensor:
        self._require_non_empty()

        return torch.stack([
            transition.state
            for transition in self.transitions
        ])

    def next_observations(self) -> Tensor:
        self._require_non_empty()

        return torch.stack([
            transition.next_state
            for transition in self.transitions
        ])

    def actions(self) -> Tensor:
        self._require_non_empty()

        return torch.tensor(
            [
                transition.action
                for transition in self.transitions
            ],
            dtype=torch.long,
        )

    def rewards(self) -> Tensor:
        self._require_non_empty()

        return torch.tensor(
            [
                transition.reward
                for transition in self.transitions
            ],
            dtype=torch.float32,
        )

    def terminated(self) -> Tensor:
        self._require_non_empty()

        return torch.tensor(
            [
                transition.terminated
                for transition in self.transitions
            ],
            dtype=torch.bool,
        )

    def log_probabilities(self) -> Tensor:
        self._require_non_empty()

        return torch.tensor(
            [
                transition.log_probability
                for transition in self.transitions
            ],
            dtype=torch.float32,
        )

    def values(self) -> Tensor:
        self._require_non_empty()

        return torch.tensor(
            [
                transition.value
                for transition in self.transitions
            ],
            dtype=torch.float32,
        )

    def elapsed_times(self) -> Tensor:
        self._require_non_empty()

        return torch.tensor(
            [
                transition.elapsed_time
                for transition in self.transitions
            ],
            dtype=torch.float32,
        )

    def _require_non_empty(self) -> None:
        if self.is_empty:
            raise ValueError(
                "Rollout is empty."
            )

    def to_tensors(self) -> RolloutTensors:
        """Convert all collected transitions into tensors."""

        if not self.transitions:
            raise ValueError(
                "Cannot convert an empty rollout."
            )

        return RolloutTensors(
            states=torch.stack(
                [transition.state for transition in self.transitions]
            ),

            actions=torch.tensor(
                [
                    transition.action
                    for transition in self.transitions
                ],
                dtype=torch.long,
            ),

            rewards=torch.tensor(
                [
                    transition.reward
                    for transition in self.transitions
                ],
                dtype=torch.float32,
            ),

            next_states=torch.stack(
                [
                    transition.next_state
                    for transition in self.transitions
                ]
            ),

            old_log_probabilities=torch.stack(
                [
                    transition.log_probability
                    for transition in self.transitions
                ]
            ).float(),

            values=torch.stack(
                [
                    transition.value
                    for transition in self.transitions
                ]
            ).float(),

            terminated=torch.tensor(
                [
                    transition.terminated
                    for transition in self.transitions
                ],
                dtype=torch.float32,
            ),

            elapsed_times=torch.tensor(
                [
                    transition.elapsed_time
                    for transition in self.transitions
                ],
                dtype=torch.float32,
            ),
        )