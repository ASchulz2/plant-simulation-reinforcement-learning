from __future__ import annotations

from dataclasses import dataclass

import torch

from environment.plant_environment import PlantEnvironment
from environment.rollout import Rollout
from learning.rollout_collector import RolloutCollector


@dataclass
class EpisodeSampler:
    """
    Manage multiple rollouts belonging to one Plant Simulation episode.

    The sampler is concerned only with episode lifecycle and rollout
    segmentation. Agent and policy semantics belong to the collector
    and agent layers.
    """

    environment: PlantEnvironment
    collector: RolloutCollector

    observation: torch.Tensor | None = None

    terminated: bool = False
    failed: bool = False
    succeeded: bool = False

    rollout_index: int = 0

    def start_episode(self) -> torch.Tensor:
        """Reset and start a new manufacturing episode."""

        self.observation = (
            self.environment.reset()
        )

        self.terminated = False
        self.failed = False
        self.succeeded = False
        self.rollout_index = 0

        return self.observation

    def sample_rollout(self) -> Rollout:
        """Sample the next fixed-horizon rollout."""

        if self.observation is None:
            raise RuntimeError(
                "Episode has not been started. "
                "Call start_episode() first."
            )

        if self.terminated:
            raise RuntimeError(
                "Episode has already terminated."
            )

        (
            rollout,
            next_observation,
            terminated,
            progress_failure,
        ) = self.collector.collect(
            self.observation
        )

        self.observation = next_observation
        self.terminated = terminated

        if terminated:

            self.failed = progress_failure

            self.succeeded = (
                not progress_failure
            )

        self.rollout_index += 1

        return rollout

    def sample_full_rollout(self) -> Rollout:
        """
        Collect until the entire manufacturing episode terminates.
        """

        if self.observation is None:
            raise RuntimeError(
                "Episode has not been started. "
                "Call start_episode() first."
            )

        if self.terminated:
            raise RuntimeError(
                "Episode has already terminated."
            )

        segments: list[Rollout] = []

        while not self.terminated:

            segments.append(
                self.sample_rollout()
            )

        total_transitions = sum(
            len(segment.transitions)
            for segment in segments
        )

        full_rollout = Rollout(
            horizon=max(
                total_transitions,
                1,
            )
        )

        for segment in segments:

            for transition in segment.transitions:

                full_rollout.add(
                    transition
                )

        return full_rollout

    def reset_episode(self) -> None:
        """Clear the current episode state."""

        self.observation = None
        self.terminated = False
        self.failed = False
        self.succeeded = False
        self.rollout_index = 0

    @property
    def episode_finished(self) -> bool:
        return self.terminated

    @property
    def episode_failed(self) -> bool:
        return self.failed

    @property
    def episode_succeeded(self) -> bool:
        return self.succeeded