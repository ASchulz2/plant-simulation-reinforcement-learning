from __future__ import annotations

import pytest

from problem.observation import Observation
from problem.reward import (
    QLearningReward,
    QLearningRewardSettings,
    RewardFunction,
)
from environment.emptying_event import EmptyingEvent
from environment.return_event import ReturnEvent

from dataclasses import replace

from environment.emptying_event import EmptyingEvent

def _make_observation(
    *,
    simulation_time: float = 1.0,
    throughput: int = 0,
    admission_buffer_contents: tuple[str, ...] = (
        "GER",
        "FR",
    ),
    storage_a_count: int = 0,
    storage_a_language: str = "",
    storage_b_count: int = 0,
    storage_b_language: str = "",
) -> Observation:
    """Create a deterministic semantic warehouse observation."""

    return Observation(
        simulation_time=simulation_time,
        admission_buffer_contents=admission_buffer_contents,
        storage_a_count=storage_a_count,
        storage_a_language=storage_a_language,
        storage_b_count=storage_b_count,
        storage_b_language=storage_b_language,
        throughput=throughput,
    )


def test_q_learning_reward_implements_reward_function() -> None:
    reward = QLearningReward()

    assert isinstance(
        reward,
        RewardFunction,
    )


def test_q_learning_reward_uses_throughput_progress() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=2.0,
            time_penalty=0.0,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation(
        throughput=10,
    )

    next_state = _make_observation(
        throughput=13,
    )

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
    )

    assert reward == pytest.approx(6.0)


def test_q_learning_reward_penalizes_elapsed_time() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            time_penalty=0.5,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation(
        simulation_time=10.0,
    )

    next_state = _make_observation(
        simulation_time=14.0,
    )

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
    )

    assert reward == pytest.approx(-2.0)


def test_q_learning_reward_combines_throughput_and_time() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=2.0,
            time_penalty=0.5,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation(
        simulation_time=10.0,
        throughput=10,
    )

    next_state = _make_observation(
        simulation_time=12.0,
        throughput=13,
    )

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
    )

    # Throughput: +6
    # Time:       -1
    # Total:       5
    assert reward == pytest.approx(5.0)


def test_q_learning_reward_success_bonus() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            time_penalty=0.0,
            success_bonus=25.0,
            failure_penalty=10.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        terminated=True,
        succeeded=True,
    )

    assert reward == pytest.approx(25.0)


def test_q_learning_reward_failure_penalty() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            time_penalty=0.0,
            success_bonus=25.0,
            failure_penalty=15.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        terminated=True,
        succeeded=False,
    )

    assert reward == pytest.approx(-15.0)


def test_q_learning_reward_does_not_apply_terminal_reward_before_termination() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            time_penalty=0.0,
            success_bonus=25.0,
            failure_penalty=15.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        terminated=False,
        succeeded=False,
    )

    assert reward == pytest.approx(0.0)


def test_q_learning_reward_ignores_succeeded_when_not_terminated() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            time_penalty=0.0,
            success_bonus=25.0,
            failure_penalty=15.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        terminated=False,
        succeeded=True,
    )

    assert reward == pytest.approx(0.0)


def test_q_learning_reward_rejects_decreasing_throughput() -> None:
    reward_function = QLearningReward()

    previous = _make_observation(
        throughput=10,
    )

    next_state = _make_observation(
        throughput=9,
    )

    with pytest.raises(
        ValueError,
        match="Throughput decreased",
    ):
        reward_function.calculate(
            previous,
            next_state,
            action=0,
        )


def test_q_learning_reward_rejects_decreasing_simulation_time() -> None:
    reward_function = QLearningReward()

    previous = _make_observation(
        simulation_time=10.0,
    )

    next_state = _make_observation(
        simulation_time=9.0,
    )

    with pytest.raises(
        ValueError,
        match="Simulation time decreased",
    ):
        reward_function.calculate(
            previous,
            next_state,
            action=0,
        )


def test_q_learning_reward_zero_progress_and_zero_time() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=1.0,
            time_penalty=0.01,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation(
        simulation_time=10.0,
        throughput=5,
    )

    next_state = _make_observation(
        simulation_time=10.0,
        throughput=5,
    )

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
    )

    assert reward == pytest.approx(0.0)


def test_q_learning_reward_applies_normalized_return_waiting_time_penalty() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            storage_reward=0.0,
            time_penalty=0.0,
            return_waiting_time_penalty=2.0,
            return_waiting_time_reference=40.0,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    events = (
        ReturnEvent(
            event_id=1,
            language="GER",
            waiting_time=10.0,
        ),
    )

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        return_events=events,
    )

    # Normalized waiting time: 10 / 40 = 0.25
    # Penalty: 2 * 0.25 = 0.5
    assert reward == pytest.approx(-0.5)

def test_q_learning_reward_applies_premature_emptying_penalty() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            storage_reward=0.0,
            time_penalty=0.0,
            return_waiting_time_penalty=0.0,
            success_bonus=0.0,
            failure_penalty=0.0,
            premature_emptying_penalty=10.0,
        ),
        storage_capacity=10,
    )

    previous = _make_observation()
    next_state = _make_observation()

    events = (
        EmptyingEvent(
            storage="A",
            event_id=1,
            mode="Drain",
            count=3,
            total_waiting_time=100.0,
        ),
    )

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        emptying_events=events,
    )

    assert reward == pytest.approx(-7.0)


def test_q_learning_reward_combines_return_events() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            storage_reward=0.0,
            time_penalty=0.0,
            return_waiting_time_penalty=1.0,
            return_waiting_time_reference=36.0,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    events = (
        ReturnEvent(
            event_id=1,
            language="GER",
            waiting_time=10.0,
        ),
        ReturnEvent(
            event_id=2,
            language="FR",
            waiting_time=7.5,
        ),
    )

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        return_events=events,
    )

    # Total waiting time: 17.5 s
    # Normalized waiting time: 17.5 / 36
    # Return penalty: -(17.5 / 36)
    assert reward == pytest.approx(
        -17.5 / 36.0
    )


def test_q_learning_reward_combines_throughput_time_and_return_waiting() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=2.0,
            storage_reward=0.0,
            time_penalty=0.5,
            return_waiting_time_penalty=0.1,
            return_waiting_time_reference=36.0,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation(
        simulation_time=10.0,
        throughput=10,
    )

    next_state = _make_observation(
        simulation_time=12.0,
        throughput=13,
    )

    return_events = (
        ReturnEvent(
            event_id=1,
            language="GER",
            waiting_time=10.0,
        ),
        ReturnEvent(
            event_id=2,
            language="FR",
            waiting_time=5.0,
        ),
    )

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        return_events=return_events,
    )

    # Throughput: +6
    # Time:       -1
    # Return waiting:
    #   -(0.1 * (15 / 36)) = -0.0416667
    # Total: 4.9583333
    assert reward == pytest.approx(
        6.0
        - 1.0
        - 0.1 * (15.0 / 36.0)
    )

@pytest.mark.parametrize(
    "field,value",
    [
        ("throughput_weight", -1.0),
        ("time_penalty", -1.0),
        ("return_waiting_time_penalty", -1.0),
        ("success_bonus", -1.0),
        ("failure_penalty", -1.0),
    ],
)
def test_reward_settings_reject_negative_values(
    field: str,
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=field,
    ):
        QLearningRewardSettings(
            **{field: value},
        )





def test_premature_emptying_penalty_is_maximum_for_empty_drain() -> None:
    reward = QLearningReward(
        QLearningRewardSettings(
            premature_emptying_penalty=10.0,
        ),
        storage_capacity=10,
    )

    event = EmptyingEvent(
        storage="StorageA",
        event_id=1,
        mode="Drain",
        count=0,
        total_waiting_time=0.0,
    )

    assert reward._premature_emptying_penalty(
        event,
        storage_capacity=10,
    ) == pytest.approx(10.0)


def test_premature_emptying_penalty_decreases_with_count() -> None:
    reward = QLearningReward(
        QLearningRewardSettings(
            premature_emptying_penalty=10.0,
        ),
        storage_capacity=10,
    )

    event = EmptyingEvent(
        storage="StorageA",
        event_id=1,
        mode="Drain",
        count=5,
        total_waiting_time=0.0,
    )

    assert reward._premature_emptying_penalty(
        event,
        storage_capacity=10,
    ) == pytest.approx(5.0)


def test_full_drain_has_no_premature_emptying_penalty() -> None:
    reward = QLearningReward(
        QLearningRewardSettings(
            premature_emptying_penalty=10.0,
        ),
        storage_capacity=10,
    )

    event = EmptyingEvent(
        storage="StorageA",
        event_id=1,
        mode="Drain",
        count=10,
        total_waiting_time=0.0,
    )

    assert reward._premature_emptying_penalty(
        event,
        storage_capacity=10,
    ) == pytest.approx(0.0)


def test_return_emptying_event_has_no_premature_emptying_penalty() -> None:
    reward = QLearningReward(
        QLearningRewardSettings(
            premature_emptying_penalty=10.0,
        ),
        storage_capacity=10,
    )

    event = EmptyingEvent(
        storage="StorageA",
        event_id=1,
        mode="Return",
        count=2,
        total_waiting_time=0.0,
    )

    assert reward._premature_emptying_penalty(
        event,
        storage_capacity=10,
    ) == pytest.approx(0.0)


def test_negative_emptying_count_is_rejected() -> None:
    reward = QLearningReward(
        QLearningRewardSettings(
            premature_emptying_penalty=10.0,
        ),
        storage_capacity=10,
    )

    event = EmptyingEvent(
        storage="StorageA",
        event_id=1,
        mode="Drain",
        count=-1,
        total_waiting_time=0.0,
    )

    with pytest.raises(
        ValueError,
        match="count must be non-negative",
    ):
        reward._premature_emptying_penalty(
            event,
            storage_capacity=10,
        )


def test_emptying_count_above_capacity_is_rejected() -> None:
    reward = QLearningReward(
        QLearningRewardSettings(
            premature_emptying_penalty=10.0,
        ),
        storage_capacity=10,
    )

    event = EmptyingEvent(
        storage="StorageA",
        event_id=1,
        mode="Drain",
        count=11,
        total_waiting_time=0.0,
    )

    with pytest.raises(
        ValueError,
        match="count must not exceed storage capacity",
    ):
        reward._premature_emptying_penalty(
            event,
            storage_capacity=10,
        )

def test_q_learning_reward_penalizes_return_action() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            time_penalty=0.0,
            return_action_penalty=3.0,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    reward = reward_function.calculate(
        previous,
        next_state,
        action=2,
    )

    assert reward == pytest.approx(-3.0)

def test_q_learning_reward_does_not_penalize_storage_actions() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            time_penalty=0.0,
            return_action_penalty=3.0,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    reward_a = reward_function.calculate(
        previous,
        next_state,
        action=0,
    )

    reward_b = reward_function.calculate(
        previous,
        next_state,
        action=1,
    )

    assert reward_a == pytest.approx(0.0)
    assert reward_b == pytest.approx(0.0)

def test_q_learning_reward_combines_return_action_and_waiting_penalties() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            time_penalty=0.0,
            return_action_penalty=3.0,
            return_waiting_time_penalty=0.5,
            success_bonus=0.0,
            failure_penalty=0.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    return_events = (
        ReturnEvent(
            event_id=1,
            language="GER",
            waiting_time=10.0,
        ),
    )

    reward = reward_function.calculate(
        previous,
        next_state,
        action=2,
        return_events=return_events,
    )

    # Action 2:          -3.0
    # Return waiting:    -5.0
    # Total:              -8.0
    assert reward == pytest.approx(-3.138888888888889)

def test_q_learning_reward_feasible_action_gets_positive_reward() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            storage_reward=0.0,
            time_penalty=0.0,
            return_waiting_time_penalty=0.0,
            success_bonus=0.0,
            failure_penalty=0.0,
            premature_emptying_penalty=0.0,
            return_action_penalty=0.0,
            feasibility_reward=1.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        action_feasible=True,
    )

    assert reward == pytest.approx(1.0)


def test_q_learning_reward_infeasible_action_gets_negative_reward() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            storage_reward=0.0,
            time_penalty=0.0,
            return_waiting_time_penalty=0.0,
            success_bonus=0.0,
            failure_penalty=0.0,
            premature_emptying_penalty=0.0,
            return_action_penalty=0.0,
            feasibility_reward=1.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
        action_feasible=False,
    )

    assert reward == pytest.approx(-1.0)


def test_q_learning_reward_omits_feasibility_component_when_not_provided() -> None:
    reward_function = QLearningReward(
        QLearningRewardSettings(
            throughput_weight=0.0,
            storage_reward=0.0,
            time_penalty=0.0,
            return_waiting_time_penalty=0.0,
            success_bonus=0.0,
            failure_penalty=0.0,
            premature_emptying_penalty=0.0,
            return_action_penalty=0.0,
            feasibility_reward=1.0,
        )
    )

    previous = _make_observation()
    next_state = _make_observation()

    reward = reward_function.calculate(
        previous,
        next_state,
        action=0,
    )

    assert reward == pytest.approx(0.0)


def test_q_learning_reward_rejects_invalid_action_feasibility() -> None:
    reward_function = QLearningReward()

    previous = _make_observation()
    next_state = _make_observation()

    with pytest.raises(
        TypeError,
        match="action_feasible",
    ):
        reward_function.calculate(
            previous,
            next_state,
            action=0,
            action_feasible=1,
        )