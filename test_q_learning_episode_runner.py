from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest
import torch

from environment.emptying_event import EmptyingEvent
from environment.episode import EpisodeOutcome
from environment.q_learning_episode_runner import QLearningEpisodeRunner
from environment.q_learning_metrics import (
    QLearningEpisodeMetrics,
    QLearningExperimentMetrics,
)
from environment.termination import TerminationReason
from problem.observation import Observation


def _observation(
    *,
    simulation_time: float = 1.0,
    throughput: int = 0,
) -> Observation:
    return Observation(
        simulation_time=simulation_time,
        admission_buffer_contents=("GER",),
        storage_a_count=0,
        storage_a_language="",
        storage_b_count=0,
        storage_b_language="",
        throughput=throughput,
    )


@dataclass
class FakeTransition:
    terminated: bool
    termination_reason: TerminationReason | None

    action: int = 0
    reward: float = 0.0
    next_simulation_time: float = 2.0
    next_throughput: int = 20

    # Keep the fake aligned with the real Transition contract.
    emptying_events: tuple[EmptyingEvent, ...] = ()


class FakeEnvironment:
    def __init__(
        self,
        *,
        observations: list[Observation],
        transitions: list[FakeTransition],
        decision_point_ids: list[int],
    ) -> None:
        if len(observations) != len(transitions) + 1:
            raise ValueError(
                "Expected one more observation than transitions."
            )

        if len(decision_point_ids) != len(transitions):
            raise ValueError(
                "Expected one decision-point ID per transition."
            )

        self._observations = observations
        self._transitions = transitions
        self._decision_point_ids = decision_point_ids

        self._observation_index = 0
        self._transition_index = 0

        self.reset_calls = 0
        self.stop_calls = 0
        self.actions: list[int] = []
        self.step_arguments: list[dict] = []

        self._step_count = 0

    @property
    def current_state(self) -> Observation:
        return self._observations[self._observation_index]

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def decision_point_id(self) -> int:
        if self._transition_index == 0:
            return 1

        return self._decision_point_ids[
            self._transition_index - 1
        ]

    def reset(self) -> None:
        self.reset_calls += 1
        self._observation_index = 0
        self._transition_index = 0
        self._step_count = 0

    def step(
        self,
        action: int,
        *,
        log_probability: torch.Tensor,
        value: torch.Tensor,
    ) -> FakeTransition:
        self.actions.append(action)

        self.step_arguments.append(
            {
                "action": action,
                "log_probability": log_probability,
                "value": value,
            }
        )

        transition = self._transitions[
            self._transition_index
        ]

        self._transition_index += 1
        self._observation_index += 1
        self._step_count += 1

        return transition

    def stop(self) -> None:
        self.stop_calls += 1


class FakeAgent:
    def __init__(self, actions: list[int]) -> None:
        self.actions = actions
        self.action_index = 0
        self.act_calls: list[Observation] = []
        self.learn_calls: list[dict] = []

    def act(self, observation: Observation) -> int:
        self.act_calls.append(observation)

        action = self.actions[self.action_index]
        self.action_index += 1

        return action

    def learn(
        self,
        *,
        observation: Observation,
        action: int,
        next_observation: Observation,
        transition: FakeTransition,
    ) -> float:
        self.learn_calls.append(
            {
                "observation": observation,
                "action": action,
                "next_observation": next_observation,
                "transition": transition,
            }
        )

        return 0.0


def test_run_episode_executes_terminal_transition_and_learns_from_it():
    observation = _observation(
        simulation_time=1.0,
        throughput=10,
    )

    next_observation = _observation(
        simulation_time=2.0,
        throughput=20,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=TerminationReason.SUCCESS,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[1],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    result, _ = runner.run_episode(
        episode_number=1,
    )

    assert result.episode_number == 1
    assert result.outcome is EpisodeOutcome.SUCCESS
    assert result.decision_points == 1
    assert result.final_decision_point_id == 2

    assert environment.reset_calls == 1
    assert environment.actions == [1]

    assert len(agent.act_calls) == 1
    assert agent.act_calls[0] == observation

    assert len(agent.learn_calls) == 1

    learn_call = agent.learn_calls[0]

    assert learn_call["observation"] == observation
    assert learn_call["action"] == 1
    assert learn_call["next_observation"] == next_observation
    assert learn_call["transition"] is transition


def test_run_episode_passes_neutral_ppo_values_to_environment():
    observation = _observation()

    next_observation = _observation(
        simulation_time=2.0,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=TerminationReason.HORIZON_TIMEOUT,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[0],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    runner.run_episode(
        episode_number=1,
    )

    step_arguments = environment.step_arguments[0]

    assert torch.equal(
        step_arguments["log_probability"],
        torch.tensor(0.0),
    )

    assert torch.equal(
        step_arguments["value"],
        torch.tensor(0.0),
    )


def test_run_episode_continues_until_terminal_transition():
    observation_1 = _observation(
        simulation_time=1.0,
        throughput=10,
    )

    observation_2 = _observation(
        simulation_time=2.0,
        throughput=20,
    )

    observation_3 = _observation(
        simulation_time=3.0,
        throughput=30,
    )

    transition_1 = FakeTransition(
        terminated=False,
        termination_reason=None,
    )

    transition_2 = FakeTransition(
        terminated=True,
        termination_reason=(
            TerminationReason.MANUFACTURING_FAILURE
        ),
    )

    environment = FakeEnvironment(
        observations=[
            observation_1,
            observation_2,
            observation_3,
        ],
        transitions=[
            transition_1,
            transition_2,
        ],
        decision_point_ids=[
            2,
            3,
        ],
    )

    agent = FakeAgent(
        actions=[0, 2],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    result, _ = runner.run_episode(
        episode_number=4,
    )

    assert result.episode_number == 4
    assert (
        result.outcome
        is EpisodeOutcome.MANUFACTURING_FAILURE
    )
    assert result.decision_points == 2
    assert result.final_decision_point_id == 3

    assert agent.act_calls == [
        observation_1,
        observation_2,
    ]

    assert environment.actions == [
        0,
        2,
    ]

    assert len(agent.learn_calls) == 2

    assert (
        agent.learn_calls[0]["transition"]
        is transition_1
    )

    assert (
        agent.learn_calls[1]["transition"]
        is transition_2
    )

    assert (
        agent.learn_calls[0]["next_observation"]
        == observation_2
    )

    assert (
        agent.learn_calls[1]["next_observation"]
        == observation_3
    )


@pytest.mark.parametrize(
    ("termination_reason", "expected_outcome"),
    [
        (
            TerminationReason.SUCCESS,
            EpisodeOutcome.SUCCESS,
        ),
        (
            TerminationReason.MANUFACTURING_FAILURE,
            EpisodeOutcome.MANUFACTURING_FAILURE,
        ),
        (
            TerminationReason.HORIZON_TIMEOUT,
            EpisodeOutcome.HORIZON_TIMEOUT,
        ),
        (
            TerminationReason.COMMUNICATION_FAILURE,
            EpisodeOutcome.COMMUNICATION_FAILURE,
        ),
    ],
)
def test_run_episode_maps_termination_reason_to_episode_outcome(
    termination_reason: TerminationReason,
    expected_outcome: EpisodeOutcome,
):
    observation = _observation()

    next_observation = _observation(
        simulation_time=2.0,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=termination_reason,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[0],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    result, _ = runner.run_episode(
        episode_number=1,
    )

    assert result.outcome is expected_outcome


@pytest.mark.parametrize(
    "episode_number",
    [
        0,
        -1,
    ],
)
def test_run_episode_rejects_invalid_episode_number(
    episode_number: int,
):
    environment = FakeEnvironment(
        observations=[
            _observation(),
        ],
        transitions=[],
        decision_point_ids=[],
    )

    agent = FakeAgent(
        actions=[],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    with pytest.raises(
        ValueError,
        match="episode_number must be at least 1",
    ):
        runner.run_episode(
            episode_number=episode_number,
        )


@pytest.mark.parametrize(
    "num_episodes",
    [
        0,
        -1,
    ],
)
def test_run_rejects_invalid_number_of_episodes(
    num_episodes: int,
):
    environment = FakeEnvironment(
        observations=[
            _observation(),
        ],
        transitions=[],
        decision_point_ids=[],
    )

    agent = FakeAgent(
        actions=[],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    with pytest.raises(
        ValueError,
        match="num_episodes must be at least 1",
    ):
        runner.run(
            num_episodes=num_episodes,
        )


def test_run_handles_keyboard_interrupt():
    environment = FakeEnvironment(
        observations=[
            _observation(),
        ],
        transitions=[],
        decision_point_ids=[],
    )

    agent = FakeAgent(
        actions=[],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    def interrupted_run_episode(
        episode_number: int,
    ):
        raise KeyboardInterrupt

    runner.run_episode = interrupted_run_episode

    results, _ = runner.run(
        num_episodes=5,
    )

    assert len(results) == 1

    result = results[0]

    assert result.episode_number == 1
    assert result.outcome is EpisodeOutcome.INTERRUPTED
    assert result.decision_points == 0
    assert result.final_decision_point_id == 1
    assert result.reason == "Episode interrupted by user."

    assert environment.stop_calls == 1


def test_run_stops_after_keyboard_interrupt():
    environment = FakeEnvironment(
        observations=[
            _observation(),
        ],
        transitions=[],
        decision_point_ids=[],
    )

    agent = FakeAgent(
        actions=[],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    episode_numbers: list[int] = []

    def interrupted_run_episode(
        episode_number: int,
    ):
        episode_numbers.append(episode_number)
        raise KeyboardInterrupt

    runner.run_episode = interrupted_run_episode

    runner.run(
        num_episodes=5,
    )

    assert episode_numbers == [1]


def test_run_episode_rejects_terminal_transition_without_reason():
    observation = _observation()

    next_observation = _observation(
        simulation_time=2.0,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=None,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[1],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    with pytest.raises(
        RuntimeError,
        match="Terminal transition has no termination reason",
    ):
        runner.run_episode(
            episode_number=1,
        )


def test_run_episode_returns_metrics():
    observation = _observation(
        simulation_time=1.0,
        throughput=10,
    )

    next_observation = _observation(
        simulation_time=2.0,
        throughput=20,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=TerminationReason.SUCCESS,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[1],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    result, metrics = runner.run_episode(
        episode_number=1,
    )

    assert result.outcome is EpisodeOutcome.SUCCESS

    assert metrics.episode_number == 1
    assert metrics.outcome is EpisodeOutcome.SUCCESS
    assert metrics.decision_points == 1
    assert metrics.simulation_time == 2.0
    assert metrics.throughput == 20

    assert metrics.drain_event_count == 0
    assert metrics.drained_mu_count == 0


def test_run_episode_aggregates_emptying_events_into_metrics():
    observation = _observation(
        simulation_time=1.0,
        throughput=10,
    )

    next_observation = _observation(
        simulation_time=2.0,
        throughput=20,
    )

    emptying_event_a = EmptyingEvent(
        storage="A",
        event_id=1,
        mode="Drain",
        count=10,
        total_waiting_time=100.0,
    )

    emptying_event_b = EmptyingEvent(
        storage="B",
        event_id=1,
        mode="Drain",
        count=7,
        total_waiting_time=70.0,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=TerminationReason.SUCCESS,
        emptying_events=(
            emptying_event_a,
            emptying_event_b,
        ),
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[0],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    _, metrics = runner.run_episode(
        episode_number=1,
    )

    assert metrics.drain_event_count == 2
    assert metrics.drained_mu_count == 17
    assert metrics.mean_drain_count == pytest.approx(8.5)


def test_run_accumulates_episode_metrics():
    observation = _observation(
        simulation_time=1.0,
        throughput=10,
    )

    next_observation = _observation(
        simulation_time=2.0,
        throughput=20,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=TerminationReason.SUCCESS,
        reward=5.0,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[1, 1],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    results, metrics = runner.run(
        num_episodes=2,
    )

    assert len(results) == 2
    assert metrics.count == 2

    assert metrics.episodes[0].episode_number == 1
    assert metrics.episodes[1].episode_number == 2

    assert metrics.mean_reward == 5.0
    assert metrics.mean_throughput == 20.0


def test_run_calls_episode_complete_callback() -> None:
    observation = _observation()

    next_observation = _observation(
        simulation_time=2.0,
        throughput=20,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=TerminationReason.SUCCESS,
        reward=5.0,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[1],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    callbacks: list[tuple] = []

    def on_episode_complete(
        episode_number: int,
        result,
        episode_metrics: QLearningEpisodeMetrics,
        cumulative_decision_count: int,
    ) -> None:
        callbacks.append(
            (
                episode_number,
                result,
                episode_metrics,
                cumulative_decision_count,
            )
        )

    results, metrics = runner.run(
        num_episodes=1,
        on_episode_complete=on_episode_complete,
    )

    assert len(callbacks) == 1

    (
        episode_number,
        result,
        episode_metrics,
        cumulative_count,
    ) = callbacks[0]

    assert episode_number == 1
    assert result is results[0]
    assert episode_metrics is metrics.episodes[0]
    assert cumulative_count == 1


def test_run_callback_receives_cumulative_decision_count() -> None:
    observation = _observation()

    next_observation = _observation(
        simulation_time=2.0,
        throughput=20,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=TerminationReason.SUCCESS,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[0, 1],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    cumulative_counts: list[int] = []

    def on_episode_complete(
        episode_number: int,
        result,
        episode_metrics: QLearningEpisodeMetrics,
        cumulative_decision_count: int,
    ) -> None:
        cumulative_counts.append(
            cumulative_decision_count
        )

    runner.run(
        num_episodes=2,
        on_episode_complete=on_episode_complete,
    )

    assert cumulative_counts == [1, 2]


def test_run_supports_resumed_episode_numbering() -> None:
    observation = _observation()

    next_observation = _observation(
        simulation_time=2.0,
        throughput=20,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=TerminationReason.SUCCESS,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[0, 1],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    results, metrics = runner.run(
        num_episodes=2,
        start_episode=11,
    )

    assert [
        item.episode_number
        for item in results
    ] == [
        11,
        12,
    ]

    assert [
        item.episode_number
        for item in metrics.episodes
    ] == [
        11,
        12,
    ]


def test_run_supports_initial_decision_count() -> None:
    observation = _observation()

    next_observation = _observation(
        simulation_time=2.0,
        throughput=20,
    )

    transition = FakeTransition(
        terminated=True,
        termination_reason=TerminationReason.SUCCESS,
    )

    environment = FakeEnvironment(
        observations=[
            observation,
            next_observation,
        ],
        transitions=[
            transition,
        ],
        decision_point_ids=[
            2,
        ],
    )

    agent = FakeAgent(
        actions=[0],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    cumulative_counts: list[int] = []

    def on_episode_complete(
        episode_number: int,
        result,
        episode_metrics: QLearningEpisodeMetrics,
        cumulative_decision_count: int,
    ) -> None:
        cumulative_counts.append(
            cumulative_decision_count
        )

    runner.run(
        num_episodes=1,
        initial_decision_count=347,
        on_episode_complete=on_episode_complete,
    )

    assert cumulative_counts == [348]


@pytest.mark.parametrize(
    "start_episode",
    [0, -1],
)
def test_run_rejects_invalid_start_episode(
    start_episode: int,
) -> None:
    environment = FakeEnvironment(
        observations=[
            _observation(),
        ],
        transitions=[],
        decision_point_ids=[],
    )

    agent = FakeAgent(
        actions=[],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    with pytest.raises(
        ValueError,
        match="start_episode must be at least 1",
    ):
        runner.run(
            num_episodes=1,
            start_episode=start_episode,
        )


@pytest.mark.parametrize(
    "initial_decision_count",
    [-1, -10],
)
def test_run_rejects_negative_initial_decision_count(
    initial_decision_count: int,
) -> None:
    environment = FakeEnvironment(
        observations=[
            _observation(),
        ],
        transitions=[],
        decision_point_ids=[],
    )

    agent = FakeAgent(
        actions=[],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    with pytest.raises(
        ValueError,
        match="initial_decision_count must be non-negative",
    ):
        runner.run(
            num_episodes=1,
            initial_decision_count=initial_decision_count,
        )


def test_run_does_not_checkpoint_interrupted_episode() -> None:
    environment = FakeEnvironment(
        observations=[
            _observation(),
        ],
        transitions=[],
        decision_point_ids=[],
    )

    agent = FakeAgent(
        actions=[],
    )

    runner = QLearningEpisodeRunner(
        environment=environment,
        agent=agent,
    )

    def interrupted_run_episode(
        episode_number: int,
    ):
        raise KeyboardInterrupt

    runner.run_episode = interrupted_run_episode

    callbacks: list[int] = []

    def on_episode_complete(
        episode_number: int,
        result,
        episode_metrics: QLearningEpisodeMetrics,
        cumulative_decision_count: int,
    ) -> None:
        callbacks.append(episode_number)

    results, _ = runner.run(
        num_episodes=5,
        on_episode_complete=on_episode_complete,
    )

    assert callbacks == []
    assert len(results) == 1
    assert (
        results[0].outcome
        is EpisodeOutcome.INTERRUPTED
    )