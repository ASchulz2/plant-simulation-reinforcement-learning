from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from environment.episode import (
    EpisodeOutcome,
    EpisodeResult,
)
from environment.transition import Transition


@dataclass(frozen=True)
class QLearningEpisodeMetrics:
    """Metrics collected for one completed Q-learning episode."""

    episode_number: int
    outcome: EpisodeOutcome
    decision_points: int
    simulation_time: float
    throughput: int
    total_reward: float
    action_counts: tuple[int, int, int]
    drain_event_count: int = 0
    drained_mu_count: int = 0

    @property
    def storage_a_actions(self) -> int:
        """Return the number of StorageA actions."""

        return self.action_counts[0]

    @property
    def storage_b_actions(self) -> int:
        """Return the number of StorageB actions."""

        return self.action_counts[1]

    @property
    def return_actions(self) -> int:
        """Return the number of Return actions."""

        return self.action_counts[2]

    @property
    def action_fractions(self) -> tuple[float, float, float]:
        """
        Return the fraction of episode actions for each action.

        The tuple order is:
            0 = StorageA
            1 = StorageB
            2 = Return
        """

        total_actions = sum(self.action_counts)

        if total_actions == 0:
            return (0.0, 0.0, 0.0)

        return tuple(
            count / total_actions
            for count in self.action_counts
        )

    @property
    def mean_drain_count(self) -> float:
        """Return the mean number of MUs per drain event."""

        if self.drain_event_count == 0:
            return 0.0

        return (
            self.drained_mu_count
            / self.drain_event_count
        )


@dataclass
class QLearningExperimentMetrics:
    """Collection of per-episode Q-learning metrics."""

    episodes: list[QLearningEpisodeMetrics] = field(
        default_factory=list
    )

    def add(
        self,
        episode: QLearningEpisodeMetrics,
    ) -> None:
        """Add one completed episode to the experiment metrics."""

        self.episodes.append(episode)

    def record(
        self,
        *,
        result: EpisodeResult,
        transitions: Iterable[Transition],
    ) -> QLearningEpisodeMetrics:
        """Create and store metrics for one completed episode."""

        transitions = tuple(transitions)

        total_reward = sum(
            float(transition.reward)
            for transition in transitions
        )

        if transitions:
            final_transition = transitions[-1]

            simulation_time = (
                final_transition.next_simulation_time
            )

            throughput = (
                final_transition.next_throughput
            )
        else:
            simulation_time = 0.0
            throughput = 0

        action_counter = Counter(
            transition.action
            for transition in transitions
        )

        action_counts = (
            int(action_counter.get(0, 0)),
            int(action_counter.get(1, 0)),
            int(action_counter.get(2, 0)),
        )

        drain_counts = [
            event.count
            for transition in transitions
            for event in transition.emptying_events
            if event.is_drain
        ]

        drain_event_count = len(drain_counts)
        drained_mu_count = sum(drain_counts)

        metrics = QLearningEpisodeMetrics(
            episode_number=result.episode_number,
            outcome=result.outcome,
            decision_points=result.decision_points,
            simulation_time=simulation_time,
            throughput=throughput,
            total_reward=total_reward,
            action_counts=action_counts,
            drain_event_count=drain_event_count,
            drained_mu_count=drained_mu_count,
        )

        self.episodes.append(metrics)

        return metrics

    @property
    def count(self) -> int:
        return len(self.episodes)

    @property
    def total_decision_points(self) -> int:
        return sum(
            episode.decision_points
            for episode in self.episodes
        )

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0

        return sum(
            episode.total_reward
            for episode in self.episodes
        ) / len(self.episodes)

    @property
    def mean_throughput(self) -> float:
        if not self.episodes:
            return 0.0

        return sum(
            episode.throughput
            for episode in self.episodes
        ) / len(self.episodes)

    @property
    def total_action_counts(self) -> tuple[int, int, int]:
        """Return action counts accumulated across all episodes."""

        return (
            sum(
                episode.action_counts[0]
                for episode in self.episodes
            ),
            sum(
                episode.action_counts[1]
                for episode in self.episodes
            ),
            sum(
                episode.action_counts[2]
                for episode in self.episodes
            ),
        )

    @property
    def action_fractions(
        self,
    ) -> tuple[float, float, float]:
        """
        Return the aggregate fraction of each action
        across all recorded episodes.

        The tuple order is:
            0 = StorageA
            1 = StorageB
            2 = Return
        """

        total_actions = sum(
            self.total_action_counts
        )

        if total_actions == 0:
            return (0.0, 0.0, 0.0)

        return tuple(
            count / total_actions
            for count in self.total_action_counts
        )

    @property
    def mean_action_counts(
        self,
    ) -> tuple[float, float, float]:
        """Return mean action counts per episode."""

        if not self.episodes:
            return (0.0, 0.0, 0.0)

        count = len(self.episodes)

        return (
            sum(
                episode.action_counts[0]
                for episode in self.episodes
            ) / count,
            sum(
                episode.action_counts[1]
                for episode in self.episodes
            ) / count,
            sum(
                episode.action_counts[2]
                for episode in self.episodes
            ) / count,
        )

    @property
    def total_drain_event_count(self) -> int:
        """Return the total number of drain events."""

        return sum(
            episode.drain_event_count
            for episode in self.episodes
        )

    @property
    def total_drained_mu_count(self) -> int:
        """Return the total number of MUs drained."""

        return sum(
            episode.drained_mu_count
            for episode in self.episodes
        )

    @property
    def mean_drain_count(self) -> float:
        """Return the mean number of MUs per drain event."""

        if self.total_drain_event_count == 0:
            return 0.0

        return (
            self.total_drained_mu_count
            / self.total_drain_event_count
        )

    @property
    def simulation_time(self) -> float:
        """Return the simulation time of the last episode."""
        return self.episodes[-1].simulation_time