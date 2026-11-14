from __future__ import annotations

from environment.emptying_event import EmptyingEvent
from environment.episode import (
    EpisodeOutcome,
    EpisodeResult,
)
from environment.q_learning_metrics import (
    QLearningEpisodeMetrics,
    QLearningExperimentMetrics,
)
from environment.transition import Transition
from problem.observation import Observation


def _observation(
    *,
    simulation_time: float,
    throughput: int,
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


def _transition(
    *,
    action: int = 0,
    reward: float,
    simulation_time: float,
    throughput: int,
    terminated: bool = False,
    emptying_events: tuple[EmptyingEvent, ...] = (),
) -> Transition:
    state = _observation(
        simulation_time=simulation_time - 1.0,
        throughput=max(0, throughput - 1),
    )

    next_state = _observation(
        simulation_time=simulation_time,
        throughput=throughput,
    )

    return Transition(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        log_probability=None,
        value=None,
        terminated=terminated,
        elapsed_time=1.0,
        simulation_time=simulation_time - 1.0,
        next_simulation_time=simulation_time,
        throughput=state.throughput,
        next_throughput=next_state.throughput,
        emptying_events=emptying_events,
        progress_failure=False,
        succeeded=False,
        termination_reason=None,
    )


def test_record_creates_episode_metrics() -> None:
    metrics = QLearningExperimentMetrics()

    result = EpisodeResult(
        episode_number=1,
        outcome=EpisodeOutcome.HORIZON_TIMEOUT,
        decision_points=3,
        final_decision_point_id=3,
    )

    transitions = [
        _transition(
            action=0,
            reward=1.5,
            simulation_time=1.0,
            throughput=1,
        ),
        _transition(
            action=1,
            reward=2.0,
            simulation_time=2.0,
            throughput=2,
        ),
        _transition(
            action=2,
            reward=-0.5,
            simulation_time=3.0,
            throughput=3,
            terminated=True,
        ),
    ]

    recorded = metrics.record(
        result=result,
        transitions=transitions,
    )

    assert isinstance(
        recorded,
        QLearningEpisodeMetrics,
    )

    assert recorded.episode_number == 1

    assert (
        recorded.outcome
        is EpisodeOutcome.HORIZON_TIMEOUT
    )

    assert recorded.decision_points == 3
    assert recorded.simulation_time == 3.0
    assert recorded.throughput == 3
    assert recorded.total_reward == 3.0

    assert recorded.action_counts == (1, 1, 1)

    assert recorded.storage_a_actions == 1
    assert recorded.storage_b_actions == 1
    assert recorded.return_actions == 1

    assert recorded.mean_drain_count == 0.0


def test_record_counts_repeated_actions() -> None:
    metrics = QLearningExperimentMetrics()

    result = EpisodeResult(
        episode_number=1,
        outcome=EpisodeOutcome.SUCCESS,
        decision_points=6,
        final_decision_point_id=6,
    )

    transitions = [
        _transition(
            action=2,
            reward=1.0,
            simulation_time=1.0,
            throughput=1,
        ),
        _transition(
            action=2,
            reward=1.0,
            simulation_time=2.0,
            throughput=2,
        ),
        _transition(
            action=0,
            reward=1.0,
            simulation_time=3.0,
            throughput=3,
        ),
        _transition(
            action=2,
            reward=1.0,
            simulation_time=4.0,
            throughput=4,
        ),
        _transition(
            action=1,
            reward=1.0,
            simulation_time=5.0,
            throughput=5,
        ),
        _transition(
            action=0,
            reward=1.0,
            simulation_time=6.0,
            throughput=6,
            terminated=True,
        ),
    ]

    recorded = metrics.record(
        result=result,
        transitions=transitions,
    )

    assert recorded.action_counts == (2, 1, 3)

    assert recorded.storage_a_actions == 2
    assert recorded.storage_b_actions == 1
    assert recorded.return_actions == 3

    assert recorded.action_fractions == (
        2 / 6,
        1 / 6,
        3 / 6,
    )


def test_experiment_metrics_accumulate_episodes() -> None:
    metrics = QLearningExperimentMetrics()

    result_1 = EpisodeResult(
        episode_number=1,
        outcome=EpisodeOutcome.HORIZON_TIMEOUT,
        decision_points=2,
        final_decision_point_id=2,
    )

    result_2 = EpisodeResult(
        episode_number=2,
        outcome=EpisodeOutcome.SUCCESS,
        decision_points=4,
        final_decision_point_id=4,
    )

    metrics.record(
        result=result_1,
        transitions=[
            _transition(
                action=0,
                reward=1.0,
                simulation_time=1.0,
                throughput=1,
                terminated=True,
            )
        ],
    )

    metrics.record(
        result=result_2,
        transitions=[
            _transition(
                action=1,
                reward=2.0,
                simulation_time=2.0,
                throughput=2,
            )
        ],
    )

    assert metrics.count == 2
    assert metrics.total_decision_points == 6
    assert metrics.mean_reward == 1.5
    assert metrics.mean_throughput == 1.5

    assert metrics.total_action_counts == (1, 1, 0)
    assert metrics.mean_action_counts == (0.5, 0.5, 0.0)

    assert metrics.total_action_counts == (1, 1, 0)

    assert metrics.action_fractions == (
        0.5,
        0.5,
        0.0,
    )


def test_empty_experiment_metrics_have_zero_means() -> None:
    metrics = QLearningExperimentMetrics()

    assert metrics.count == 0
    assert metrics.total_decision_points == 0
    assert metrics.mean_reward == 0.0
    assert metrics.mean_throughput == 0.0
    assert metrics.mean_drain_count == 0.0

    assert metrics.total_action_counts == (0, 0, 0)
    assert metrics.mean_action_counts == (
        0.0,
        0.0,
        0.0,
    )
    assert metrics.action_fractions == (
        0.0,
        0.0,
        0.0,
    )
    assert metrics.total_drain_event_count == 0
    assert metrics.total_drained_mu_count == 0


def test_add_appends_episode_metrics() -> None:
    metrics = QLearningExperimentMetrics()

    episode = QLearningEpisodeMetrics(
        episode_number=1,
        outcome=EpisodeOutcome.SUCCESS,
        decision_points=5,
        simulation_time=12.0,
        throughput=10,
        total_reward=7.5,
        action_counts=(1, 2, 2),
    )

    metrics.add(episode)

    assert metrics.count == 1
    assert metrics.episodes[0] is episode

def test_record_calculates_mean_drain_count() -> None:
    metrics = QLearningExperimentMetrics()

    result = EpisodeResult(
        episode_number=1,
        outcome=EpisodeOutcome.HORIZON_TIMEOUT,
        decision_points=2,
        final_decision_point_id=2,
    )

    transitions = [
        _transition(
            action=0,
            reward=1.0,
            simulation_time=1.0,
            throughput=1,
            emptying_events=(
                EmptyingEvent(
                    storage="A",
                    event_id=1,
                    mode="Drain",
                    count=4,
                    total_waiting_time=0.0,
                ),
            ),
        ),
        _transition(
            action=1,
            reward=1.0,
            simulation_time=2.0,
            throughput=2,
            terminated=True,
            emptying_events=(
                EmptyingEvent(
                    storage="B",
                    event_id=2,
                    mode="Drain",
                    count=8,
                    total_waiting_time=0.0,
                ),
            ),
        ),
    ]

    recorded = metrics.record(
        result=result,
        transitions=transitions,
    )

    assert recorded.drain_event_count == 2
    assert recorded.drained_mu_count == 12
    assert recorded.mean_drain_count == 6.0

    assert metrics.total_drain_event_count == 2
    assert metrics.total_drained_mu_count == 12
    assert metrics.mean_drain_count == 6.0

def test_experiment_mean_drain_count_is_event_weighted() -> None:
    metrics = QLearningExperimentMetrics()

    result_1 = EpisodeResult(
        episode_number=1,
        outcome=EpisodeOutcome.HORIZON_TIMEOUT,
        decision_points=3,
        final_decision_point_id=3,
    )

    result_2 = EpisodeResult(
        episode_number=2,
        outcome=EpisodeOutcome.HORIZON_TIMEOUT,
        decision_points=3,
        final_decision_point_id=3,
    )

    metrics.record(
        result=result_1,
        transitions=[
            _transition(
                reward=1.0,
                simulation_time=1.0,
                throughput=1,
                emptying_events=(
                    EmptyingEvent(
                        storage="A",
                        event_id=1,
                        mode="Drain",
                        count=10,
                        total_waiting_time=0.0,
                    ),
                ),
            ),
        ],
    )

    metrics.record(
        result=result_2,
        transitions=[
            _transition(
                reward=1.0,
                simulation_time=1.0,
                throughput=1,
                emptying_events=(
                    EmptyingEvent(
                        storage="A",
                        event_id=2,
                        mode="Drain",
                        count=2,
                        total_waiting_time=0.0,
                    ),
                    EmptyingEvent(
                        storage="A",
                        event_id=3,
                        mode="Drain",
                        count=2,
                        total_waiting_time=0.0,
                    ),
                ),
            ),
        ],
    )

    assert metrics.total_drain_event_count == 3
    assert metrics.total_drained_mu_count == 14
    assert metrics.mean_drain_count == 14 / 3