from __future__ import annotations

import random

import pytest

from problem.problem import ManufacturingProblem
from problem.q_learning_agent import (
    QLearningAgent,
    QLearningAgentSettings,
)
from problem.q_table import QTable
from problem.tabular_state import TabularState


def _make_agent(
    settings: QLearningAgentSettings,
) -> QLearningAgent:
    return QLearningAgent(
        q_table=QTable(),
        problem=ManufacturingProblem(),
        settings=settings,
        rng=random.Random(123),
    )


def _state_with_only_storage_a_feasible() -> TabularState:
    """Create a state in which Storage A is the only feasible storage."""
    return TabularState(
        current_language="GER",
        admission_buffer_ger_count=1,
        admission_buffer_fr_count=0,
        admission_buffer_rus_count=0,
        storage_a_language="",
        storage_a_count=0,
        storage_b_language="FR",
        storage_b_count=10,
    )

def make_state(
    *,
    current_language: str = "GER",
    storage_a_language: str = "",
    storage_a_count: int = 0,
    storage_b_language: str = "",
    storage_b_count: int = 0,
) -> TabularState:
    return TabularState(
        current_language=current_language,
        admission_buffer_ger_count=(
            1
            if current_language == "GER"
            else 0
        ),
        admission_buffer_fr_count=(
            1
            if current_language == "FR"
            else 0
        ),
        admission_buffer_rus_count=(
            1
            if current_language == "RUS"
            else 0
        ),
        storage_a_language=storage_a_language,
        storage_a_count=storage_a_count,
        storage_b_language=storage_b_language,
        storage_b_count=storage_b_count,
    )


def test_greedy_selection_chooses_highest_feasible_q_value():
    table = QTable()
    problem = ManufacturingProblem()

    state = make_state()

    table.set(state, 0, 1.0)
    table.set(state, 1, 5.0)
    table.set(state, 2, 3.0)

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )
    )

    assert agent.select_action(state) == 1


def test_greedy_selection_ignores_infeasible_actions():
    table = QTable()
    problem = ManufacturingProblem()

    state = make_state(
        storage_a_language="FR",
        storage_a_count=3,
        storage_b_language="RUS",
        storage_b_count=2,
    )

    table.set(state, 0, 100.0)
    table.set(state, 1, 100.0)
    table.set(state, 2, 1.0)

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )
    )

    assert agent.select_action(state) == 2


def test_greedy_selection_ignores_full_matching_storage():
    table = QTable()
    problem = ManufacturingProblem(
        storage_capacity=10,
    )

    state = make_state(
        current_language="GER",
        storage_a_language="GER",
        storage_a_count=10,
        storage_b_language="",
        storage_b_count=0,
    )

    table.set(state, 0, 100.0)
    table.set(state, 1, 5.0)
    table.set(state, 2, 1.0)

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )
    )

    assert agent.select_action(state) == 1


def test_greedy_selection_can_use_matching_storage_below_capacity():
    table = QTable()
    problem = ManufacturingProblem(
        storage_capacity=10,
    )

    state = make_state(
        current_language="GER",
        storage_a_language="GER",
        storage_a_count=9,
        storage_b_language="",
        storage_b_count=0,
    )

    table.set(state, 0, 100.0)
    table.set(state, 1, 5.0)
    table.set(state, 2, 1.0)

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )
    )

    assert agent.select_action(state) == 0


def test_greedy_tie_breaking_uses_first_feasible_action():
    table = QTable()
    problem = ManufacturingProblem()

    state = make_state()

    table.set(state, 0, 2.0)
    table.set(state, 1, 2.0)
    table.set(state, 2, 2.0)

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )
    )

    assert agent.select_action(state) == 0


def test_epsilon_one_always_explores():
    table = QTable()
    problem = ManufacturingProblem()

    state = make_state()

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=1.0,
            epsilon_min=1.0,
            epsilon_decay=1.0,
        ),
        rng=random.Random(123),
    )

    selected = {
        agent.select_action(state)
        for _ in range(100)
    }

    assert selected <= {0, 1, 2}
    assert selected


def test_exploration_never_selects_infeasible_actions():
    table = QTable()
    problem = ManufacturingProblem()

    state = make_state(
        storage_a_language="FR",
        storage_a_count=3,
        storage_b_language="RUS",
        storage_b_count=2,
    )

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
        ),
        rng=random.Random(123),
    )

    selected = {
        agent.select_action(state)
        for _ in range(100)
    }

    assert selected == {2}


def test_exploration_never_selects_full_matching_storage():
    table = QTable()
    problem = ManufacturingProblem(
        storage_capacity=10,
    )

    state = make_state(
        current_language="GER",
        storage_a_language="GER",
        storage_a_count=10,
        storage_b_language="",
        storage_b_count=0,
    )

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=1.0,
            epsilon_min=1.0,
            epsilon_decay=1.0,
        ),
        rng=random.Random(123),
    )

    selected = {
        agent.select_action(state)
        for _ in range(100)
    }

    assert selected == {1, 2}


def test_empty_current_language_has_no_selectable_action():
    table = QTable()
    problem = ManufacturingProblem()

    state = make_state(
        current_language=""
    )

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(),
    )

    with pytest.raises(ValueError):
        agent.select_action(state)


@pytest.mark.parametrize(
    "epsilon",
    [-0.1, 1.1],
)
def test_epsilon_is_validated(
    epsilon: float,
):
    with pytest.raises(ValueError):
        QLearningAgentSettings(
            epsilon_start=epsilon,
        )


def test_initial_epsilon() -> None:
    settings = QLearningAgentSettings()

    agent = _make_agent(settings)

    assert agent.epsilon == pytest.approx(1.0)

def test_epsilon_decays_after_action_selection() -> None:
    settings = QLearningAgentSettings(
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
    )

    agent = _make_agent(settings)

    initial = agent.epsilon

    agent.select_action(make_state())

    assert agent.epsilon < initial

def test_epsilon_does_not_fall_below_minimum() -> None:
    settings = QLearningAgentSettings(
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.5,
    )

    agent = _make_agent(settings)

    for _ in range(100):
        agent.select_action(make_state())

    assert agent.epsilon == pytest.approx(0.05)

@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon_start": -0.1},
        {"epsilon_start": 1.1},
        {"epsilon_min": -0.1},
        {"epsilon_min": 1.1},
        {"epsilon_min": 0.8, "epsilon_start": 0.5},
        {"epsilon_decay": 0.0},
        {"epsilon_decay": 1.1},
    ],
)
def test_settings_reject_invalid_epsilon_schedule(
    kwargs,
) -> None:
    with pytest.raises(ValueError):
        QLearningAgentSettings(**kwargs)

def test_snapshot_contains_selection_count() -> None:
    agent = _make_agent(
        QLearningAgentSettings(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.998,
        )
    )

    state = make_state()

    for _ in range(7):
        agent.select_action(state)

    snapshot = agent.snapshot()

    assert snapshot["selection_count"] == 7
    assert "rng_state" in snapshot


def test_restore_preserves_selection_count() -> None:
    settings = QLearningAgentSettings(
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
    )

    original = _make_agent(settings)

    for _ in range(100):
        original.select_action(make_state())

    snapshot = original.snapshot()

    restored = _make_agent(settings)

    assert restored.epsilon == pytest.approx(1.0)

    restored.restore(snapshot)

    assert restored.epsilon == pytest.approx(
        original.epsilon
    )


def test_restore_preserves_rng_sequence() -> None:
    settings = QLearningAgentSettings(
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
    )

    original = _make_agent(settings)

    for _ in range(10):
        original.select_action(make_state())

    snapshot = original.snapshot()

    restored = _make_agent(settings)

    restored.restore(snapshot)

    state = make_state()

    original_actions = [
        original.select_action(state)
        for _ in range(20)
    ]

    restored_actions = [
        restored.select_action(state)
        for _ in range(20)
    ]

    assert restored_actions == original_actions


def test_snapshot_is_independent_of_later_agent_changes() -> None:
    agent = _make_agent(
        QLearningAgentSettings(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.998,
        )
    )

    for _ in range(5):
        agent.select_action(make_state())

    snapshot = agent.snapshot()

    for _ in range(20):
        agent.select_action(make_state())

    assert snapshot["selection_count"] == 5


def test_restore_rejects_missing_selection_count() -> None:
    agent = _make_agent(
        QLearningAgentSettings()
    )

    with pytest.raises(KeyError, match="selection_count"):
        agent.restore(
            {
                "rng_state": agent.rng.getstate(),
            }
        )


def test_restore_rejects_missing_rng_state() -> None:
    agent = _make_agent(
        QLearningAgentSettings()
    )

    with pytest.raises(KeyError, match="rng_state"):
        agent.restore(
            {
                "selection_count": 10,
            }
        )


@pytest.mark.parametrize(
    "selection_count",
    [-1, -10],
)
def test_restore_rejects_negative_selection_count(
    selection_count: int,
) -> None:
    agent = _make_agent(
        QLearningAgentSettings()
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        agent.restore(
            {
                "selection_count": selection_count,
                "rng_state": agent.rng.getstate(),
            }
        )


def test_restore_rejects_non_integer_selection_count() -> None:
    agent = _make_agent(
        QLearningAgentSettings()
    )

    with pytest.raises(
        TypeError,
        match="integer",
    ):
        agent.restore(
            {
                "selection_count": 1.5,
                "rng_state": agent.rng.getstate(),
            }
        )


def test_restore_rejects_invalid_rng_state() -> None:
    agent = _make_agent(
        QLearningAgentSettings()
    )

    with pytest.raises(
        ValueError,
        match="Invalid RNG state",
    ):
        agent.restore(
            {
                "selection_count": 10,
                "rng_state": ("invalid",),
            }
        )

def test_default_agent_settings_use_feasible_actions() -> None:
    settings = QLearningAgentSettings()

    assert settings.feasible is True


def test_agent_settings_accept_infeasible_action_selection() -> None:
    settings = QLearningAgentSettings(
        feasible=False,
    )

    assert settings.feasible is False

def test_select_action_uses_feasible_actions_when_enabled() -> None:
    problem = ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        actions=(0, 1, 2),
    )

    state = _state_with_only_storage_a_feasible()

    q_table = QTable()

    agent = QLearningAgent(
        q_table=q_table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            feasible=True,
        ),
    )

    action = agent.select_action(state)

    assert action in problem.feasible_actions(state)

def test_select_action_uses_all_actions_when_feasibility_disabled() -> None:
    problem = ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        actions=(0, 1, 2),
    )

    state = _state_with_only_storage_a_feasible()

    q_table = QTable()
    q_table.set(state, 2, 10.0)

    agent = QLearningAgent(
        q_table=q_table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            feasible=False,
        ),
    )

    action = agent.select_action(state)

    assert action == 2