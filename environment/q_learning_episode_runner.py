from __future__ import annotations

from dataclasses import dataclass

import torch

from agent.tabular_q_learning_agent import TabularQLearningAgent
from environment.episode import (
    EpisodeOutcome,
    EpisodeResult,
    episode_outcome_from_termination_reason,
)
from environment.plant_environment import PlantEnvironment
from environment.q_learning_metrics import (
    QLearningEpisodeMetrics,
    QLearningExperimentMetrics,
)
from collections import Counter
from collections.abc import Callable

EpisodeCompleteCallback = Callable[
    [
        int,
        EpisodeResult,
        QLearningEpisodeMetrics,
        int,
    ],
    None,
]

@dataclass
class QLearningEpisodeRunner:
    """
    Orchestrates tabular Q-learning episodes.

    The runner deliberately contains no Q-learning logic itself.
    Action selection and learning remain inside TabularQLearningAgent.
    """

    environment: PlantEnvironment
    agent: TabularQLearningAgent
    learn: bool = True

    def run_episode(
        self,
        episode_number: int,
    ) -> tuple[
        EpisodeResult,
        QLearningEpisodeMetrics,
    ]:
        """
        Run one complete Q-learning episode.

        The terminal transition is passed to the agent before the
        episode result and metrics are returned.
        """
        if episode_number < 1:
            raise ValueError(
                "episode_number must be at least 1."
            )

        self.environment.reset()

        observation = self.environment.current_state

        if observation is None:
            raise RuntimeError(
                "Environment reset completed without a current state."
            )

        transitions = []

        while True:
            action = self.agent.act(observation)

            transition = self.environment.step(
                action,
                log_probability=torch.tensor(0.0),
                value=torch.tensor(0.0),
            )

            transitions.append(transition)

            next_observation = self.environment.current_state

            if next_observation is None:
                raise RuntimeError(
                    "Environment step completed without a current state."
                )

            if self.learn:
                self.agent.learn(
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    transition=transition,
                )

            if transition.terminated:
                termination_reason = transition.termination_reason

                if termination_reason is None:
                    raise RuntimeError(
                        "Terminal transition has no termination reason."
                    )

                outcome = episode_outcome_from_termination_reason(
                    termination_reason
                )

                result = EpisodeResult(
                    episode_number=episode_number,
                    outcome=outcome,
                    decision_points=self.environment.step_count,
                    final_decision_point_id=(
                        self.environment.decision_point_id
                    ),
                    reason=termination_reason
                )

                action_counter = Counter(
                    transition.action
                    for transition in transitions
                )

                drain_event_count = sum(
                    len(transition.emptying_events)
                    for transition in transitions
                )

                drained_mu_count = sum(
                    event.count
                    for transition in transitions
                    for event in transition.emptying_events
                )

                episode_metrics = QLearningEpisodeMetrics(
                    episode_number=episode_number,
                    outcome=outcome,
                    decision_points=result.decision_points,
                    simulation_time=float(
                        transitions[-1].next_simulation_time
                    ),
                    throughput=int(
                        transitions[-1].next_throughput
                    ),
                    total_reward=sum(
                        float(item.reward)
                        for item in transitions
                    ),
                    action_counts=(
                        int(action_counter.get(0, 0)),
                        int(action_counter.get(1, 0)),
                        int(action_counter.get(2, 0)),
                    ),
                    drain_event_count=drain_event_count,
                    drained_mu_count=drained_mu_count,
                )

                return result, episode_metrics

            observation = next_observation

    def run(
    self,
        num_episodes: int,
        *,
        start_episode: int = 1,
        initial_decision_count: int = 0,
        on_episode_complete: EpisodeCompleteCallback | None = None,
    ) -> tuple[
        list[EpisodeResult],
        QLearningExperimentMetrics,
    ]:
        """
        Run multiple Q-learning episodes.
        """
        if start_episode < 1:
            raise ValueError(
                "start_episode must be at least 1."
            )

        if initial_decision_count < 0:
            raise ValueError(
                "initial_decision_count must be non-negative."
            )

        if num_episodes < 1:
            raise ValueError(
                "num_episodes must be at least 1."
            )

        results: list[EpisodeResult] = []
        metrics = QLearningExperimentMetrics()
        cumulative_decision_count = initial_decision_count

        for episode_number in range(
            start_episode,
            start_episode + num_episodes,
        ):
            try:
                result, episode_metrics = self.run_episode(
                    episode_number
                )

            except KeyboardInterrupt:
                self.environment.stop()

                results.append(
                    EpisodeResult(
                        episode_number=episode_number,
                        outcome=EpisodeOutcome.INTERRUPTED,
                        decision_points=self.environment.step_count,
                        final_decision_point_id=(
                            self.environment.decision_point_id
                        ),
                        reason="Episode interrupted by user.",
                    )
                )

                break

            results.append(result)
            metrics.episodes.append(episode_metrics)

            cumulative_decision_count += episode_metrics.decision_points

            if on_episode_complete is not None:
                on_episode_complete(
                    episode_number,
                    result,
                    episode_metrics,
                    cumulative_decision_count,
                )

        return results, metrics


def run_tabular_q_learning_experiment(
        *,
        environment: PlantEnvironment,
        agent: TabularQLearningAgent,
        num_episodes: int,
        start_episode: int = 1,
        initial_decision_count: int = 0,
        on_episode_complete: EpisodeCompleteCallback | None = None,
    ) -> tuple[
        list[EpisodeResult],
        QLearningExperimentMetrics,
    ]:
    """
    Convenience entry point for the tabular Q-learning experiment.
    """

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    return runner.run(
        num_episodes,
        start_episode=start_episode,
        initial_decision_count=initial_decision_count,
        on_episode_complete=on_episode_complete
    )