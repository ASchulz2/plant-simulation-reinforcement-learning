from __future__ import annotations

import torch

from environment.rollout import Rollout
from learning.episode_sampler import EpisodeSampler


class FakeEnvironment:

    def __init__(self) -> None:
        self.reset_called = False

    def reset(self) -> torch.Tensor:
        self.reset_called = True

        return torch.tensor(
            [0.0],
            dtype=torch.float32,
        )


class FakeCollector:

    def __init__(self) -> None:
        self.calls = 0

    def collect(
        self,
        observation: torch.Tensor,
    ) -> tuple[
        Rollout,
        torch.Tensor,
        bool,
        bool,
    ]:

        self.calls += 1

        rollout = Rollout(
            horizon=1
        )

        next_observation = torch.tensor(
            [float(self.calls)],
            dtype=torch.float32,
        )

        return (
            rollout,
            next_observation,
            self.calls >= 2,
            False,
        )


def test_start_episode() -> None:

    environment = FakeEnvironment()
    collector = FakeCollector()

    sampler = EpisodeSampler(
        environment=environment,
        collector=collector,
    )

    observation = sampler.start_episode()

    assert environment.reset_called

    assert torch.equal(
        observation,
        torch.tensor(
            [0.0],
            dtype=torch.float32,
        ),
    )

    assert sampler.rollout_index == 0
    assert sampler.episode_finished is False
    assert sampler.episode_failed is False
    assert sampler.episode_succeeded is False


def test_sample_rollout_tracks_episode_state() -> None:

    environment = FakeEnvironment()
    collector = FakeCollector()

    sampler = EpisodeSampler(
        environment=environment,
        collector=collector,
    )

    sampler.start_episode()

    sampler.sample_rollout()

    assert sampler.rollout_index == 1
    assert sampler.episode_finished is False

    sampler.sample_rollout()

    assert sampler.rollout_index == 2
    assert sampler.episode_finished is True
    assert sampler.episode_failed is False
    assert sampler.episode_succeeded is True


def test_reset_episode() -> None:

    sampler = EpisodeSampler(
        environment=FakeEnvironment(),
        collector=FakeCollector(),
    )

    sampler.start_episode()

    sampler.sample_rollout()
    sampler.sample_rollout()

    sampler.reset_episode()

    assert sampler.observation is None
    assert sampler.episode_finished is False
    assert sampler.episode_failed is False
    assert sampler.episode_succeeded is False
    assert sampler.rollout_index == 0