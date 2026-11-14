from __future__ import annotations

import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import tabular_q_learning_main
from agent.tabular_q_learning_agent import (
    TabularQLearningAgent,
)
from checkpoint.q_learning_checkpoint import (
    QLearningCheckpointManager,
    QLearningCheckpointSettings,
    QLearningExperimentSettings,
)
from environment.transition import Transition
from problem.observation import Observation
from problem.problem import ManufacturingProblem
from problem.q_learning import (
    QLearningSettings,
    QLearningUpdater,
)
from problem.q_learning_agent import (
    QLearningAgent,
    QLearningAgentSettings,
)
from problem.q_table import QTable
from problem.tabular_q_learning import (
    TabularQLearner,
)
from problem.tabular_state import TabularState
from problem.tabular_state_encoder import (
    TabularStateEncoder,
)


def make_state(
    *,
    current_language: str = "GER",
) -> TabularState:
    return TabularState(
        current_language=current_language,
        admission_buffer_ger_count=(
            1 if current_language == "GER" else 0
        ),
        admission_buffer_fr_count=(
            1 if current_language == "FR" else 0
        ),
        admission_buffer_rus_count=(
            1 if current_language == "RUS" else 0
        ),
        storage_a_language="GER",
        storage_a_count=2,
        storage_b_language="FR",
        storage_b_count=1,
    )


def make_learner(
    q_table: QTable,
    *,
    rng: random.Random | None = None,
    feasible: bool = True,
) -> TabularQLearner:
    problem = ManufacturingProblem()

    agent = QLearningAgent(
        q_table=q_table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.998,
            feasible=feasible,
        ),
        rng=rng,
    )

    updater = QLearningUpdater(
        q_table=q_table,
        settings=QLearningSettings(
            learning_rate=0.1,
            discount_factor=0.99,
        ),
    )

    return TabularQLearner(
        agent=agent,
        updater=updater,
        problem=problem,
        feasible=feasible,
    )


def test_snapshot_contains_q_table_and_agent_state() -> None:
    q_table = QTable()
    learner = make_learner(
        q_table,
        rng=random.Random(123),
    )

    state = make_state()

    q_table.set(state, 0, 4.5)

    for _ in range(7):
        learner.select_action(state)

    snapshot = learner.snapshot()

    assert snapshot["q_table"] == {
        (state, 0): 4.5,
    }

    agent_state = snapshot["agent_state"]

    assert isinstance(agent_state, dict)
    assert agent_state["selection_count"] == 7
    assert agent_state["rng_state"] is not None


def test_restore_reconstructs_q_table_and_agent_state() -> None:
    original_table = QTable()
    original = make_learner(
        original_table,
        rng=random.Random(123),
    )

    state = make_state()

    original_table.set(state, 0, 4.5)
    original_table.set(state, 1, -2.0)

    for _ in range(25):
        original.select_action(state)

    snapshot = original.snapshot()

    restored_table = QTable()
    restored = make_learner(
        restored_table,
        rng=random.Random(999),
    )

    restored.restore(snapshot)

    assert (
        restored_table.snapshot()
        == original_table.snapshot()
    )

    assert restored.agent.epsilon == pytest.approx(
        original.agent.epsilon
    )


def test_restore_preserves_agent_rng_sequence() -> None:
    original_table = QTable()
    original = make_learner(
        original_table,
        rng=random.Random(123),
    )

    state = make_state()

    for _ in range(10):
        original.select_action(state)

    snapshot = original.snapshot()

    restored_table = QTable()
    restored = make_learner(
        restored_table,
        rng=random.Random(999),
    )

    restored.restore(snapshot)

    original_actions = [
        original.select_action(state)
        for _ in range(20)
    ]

    restored_actions = [
        restored.select_action(state)
        for _ in range(20)
    ]

    assert restored_actions == original_actions


def test_restore_replaces_existing_q_table_state() -> None:
    state = make_state()

    original_table = QTable()
    original_table.set(state, 0, 10.0)

    original = make_learner(
        original_table,
        rng=random.Random(123),
    )

    snapshot = original.snapshot()

    restored_table = QTable()
    restored_table.set(state, 1, 99.0)

    restored = make_learner(
        restored_table,
        rng=random.Random(999),
    )

    restored.restore(snapshot)

    assert restored_table.get(state, 0) == 10.0
    assert restored_table.get(state, 1) == 0.0


def test_restore_rejects_missing_q_table() -> None:
    learner = make_learner(
        QTable(),
        rng=random.Random(123),
    )

    with pytest.raises(
        KeyError,
        match="q_table",
    ):
        learner.restore(
            {
                "agent_state": learner.agent.snapshot(),
            }
        )


def test_restore_rejects_missing_agent_state() -> None:
    learner = make_learner(
        QTable(),
        rng=random.Random(123),
    )

    with pytest.raises(
        KeyError,
        match="agent_state",
    ):
        learner.restore(
            {
                "q_table": {},
            }
        )


def test_restore_rejects_invalid_q_table_snapshot() -> None:
    learner = make_learner(
        QTable(),
        rng=random.Random(123),
    )

    with pytest.raises(
        TypeError,
        match="q_table snapshot must be a dictionary",
    ):
        learner.restore(
            {
                "q_table": [],
                "agent_state": learner.agent.snapshot(),
            }
        )


def test_restore_rejects_invalid_agent_state_snapshot() -> None:
    learner = make_learner(
        QTable(),
        rng=random.Random(123),
    )

    with pytest.raises(
        TypeError,
        match="agent_state must be a dictionary",
    ):
        learner.restore(
            {
                "q_table": {},
                "agent_state": [],
            }
        )


def make_observation(
    *,
    simulation_time: float = 10.0,
    admission_buffer_contents=(
        "GER",
        "FR",
        "RUS",
    ),
    storage_a_count: int = 0,
    storage_a_language: str = "",
    storage_b_count: int = 0,
    storage_b_language: str = "",
    throughput: int = 0,
) -> Observation:
    return Observation(
        simulation_time=simulation_time,
        admission_buffer_contents=admission_buffer_contents,
        storage_a_count=storage_a_count,
        storage_a_language=storage_a_language,
        storage_b_count=storage_b_count,
        storage_b_language=storage_b_language,
        throughput=throughput,
    )


def make_agent(
    table: QTable | None = None,
    *,
    problem: ManufacturingProblem | None = None,
) -> TabularQLearningAgent:
    if table is None:
        table = QTable()

    if problem is None:
        problem = ManufacturingProblem()

    q_agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.998,
        ),
    )

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(
            learning_rate=0.5,
            discount_factor=0.9,
        ),
    )

    learner = TabularQLearner(
        agent=q_agent,
        updater=updater,
        problem=problem,
    )

    return TabularQLearningAgent(
        state_encoder=TabularStateEncoder(problem),
        learner=learner,
    )


def make_transition(
    *,
    reward: float,
    terminated: bool,
) -> Transition:
    zero = torch.tensor(0.0)

    return Transition(
        state=torch.zeros(1),
        action=0,
        reward=reward,
        next_state=torch.zeros(1),
        log_probability=zero,
        value=zero,
        terminated=terminated,
        progress_failure=False,
        succeeded=False,
        simulation_time=1.0,
        next_simulation_time=2.0,
        elapsed_time=1.0,
        throughput=0,
        next_throughput=0,
    )


def test_agent_selects_action_from_semantic_observation():
    agent = make_agent()

    observation = make_observation()

    action = agent.act(observation)

    assert action in (0, 1, 2)


def test_agent_selection_updates_state_visitation():
    agent = make_agent()

    observation = make_observation()

    agent.act(observation)
    agent.act(observation)

    state = agent.state_encoder.encode(
        observation
    )

    assert (
        agent.learner.state_registry.visits(state)
        == 2
    )


def test_agent_selection_updates_state_action_visitation():
    agent = make_agent()

    observation = make_observation()

    action = agent.act(observation)

    state = agent.state_encoder.encode(
        observation
    )

    assert (
        agent.learner.state_action_registry.visits(
            state,
            action,
        )
        == 1
    )


def test_agent_learning_updates_q_table():
    table = QTable()
    agent = make_agent(table)

    observation = make_observation(
        admission_buffer_contents=("GER",)
    )

    next_observation = make_observation(
        admission_buffer_contents=("FR",)
    )

    transition = make_transition(
        reward=10.0,
        terminated=True,
    )

    agent.learn(
        observation=observation,
        action=0,
        next_observation=next_observation,
        transition=transition,
    )

    state = agent.state_encoder.encode(
        observation
    )

    assert table.get(
        state,
        0,
    ) == pytest.approx(5.0)


def test_agent_learning_does_not_add_visitation():
    agent = make_agent()

    observation = make_observation(
        admission_buffer_contents=("GER",)
    )

    next_observation = make_observation(
        admission_buffer_contents=("FR",)
    )

    transition = make_transition(
        reward=1.0,
        terminated=True,
    )

    agent.learn(
        observation=observation,
        action=0,
        next_observation=next_observation,
        transition=transition,
    )

    state = agent.state_encoder.encode(
        observation
    )

    assert (
        agent.learner.state_registry.visits(state)
        == 0
    )

    assert (
        agent.learner.state_action_registry.visits(
            state,
            0,
        )
        == 0
    )


def test_agent_learning_excludes_full_storage_from_bootstrap():
    table = QTable()

    problem = ManufacturingProblem(
        storage_capacity=10,
    )

    agent = make_agent(
        table,
        problem=problem,
    )

    observation = make_observation(
        admission_buffer_contents=("GER",)
    )

    next_observation = make_observation(
        admission_buffer_contents=("GER",),
        storage_a_count=10,
        storage_a_language="GER",
        storage_b_count=0,
        storage_b_language="",
    )

    next_state = agent.state_encoder.encode(
        next_observation
    )

    table.set(
        next_state,
        0,
        100.0,
    )

    table.set(
        next_state,
        1,
        5.0,
    )

    table.set(
        next_state,
        2,
        1.0,
    )

    transition = make_transition(
        reward=10.0,
        terminated=False,
    )

    agent.learn(
        observation=observation,
        action=0,
        next_observation=next_observation,
        transition=transition,
    )

    state = agent.state_encoder.encode(
        observation
    )

    assert table.get(
        state,
        0,
    ) == pytest.approx(7.25)


def test_agent_learning_uses_both_capacity_and_language_constraints():
    table = QTable()

    problem = ManufacturingProblem(
        storage_capacity=10,
    )

    agent = make_agent(
        table,
        problem=problem,
    )

    observation = make_observation(
        admission_buffer_contents=("GER",)
    )

    next_observation = make_observation(
        admission_buffer_contents=("GER",),
        storage_a_count=10,
        storage_a_language="FR",
        storage_b_count=10,
        storage_b_language="RUS",
    )

    next_state = agent.state_encoder.encode(
        next_observation
    )

    table.set(
        next_state,
        0,
        100.0,
    )

    table.set(
        next_state,
        1,
        100.0,
    )

    table.set(
        next_state,
        2,
        4.0,
    )

    transition = make_transition(
        reward=2.0,
        terminated=False,
    )

    agent.learn(
        observation=observation,
        action=0,
        next_observation=next_observation,
        transition=transition,
    )

    state = agent.state_encoder.encode(
        observation
    )

    assert table.get(
        state,
        0,
    ) == pytest.approx(2.8)


def test_run_wires_checkpoint_manager_into_episode_completion(
    monkeypatch,
):
    environment = Mock()
    environment.plant.close_model.return_value = None
    environment.plant.quit.return_value = None
    environment.tcp_client.close.return_value = None

    agent = Mock()
    agent.learner.state_registry.count.return_value = 0
    agent.learner.state_action_registry.count.return_value = 0
    metrics = Mock()
    metrics.episodes = []
    metrics.mean_reward = 0.0
    metrics.mean_throughput = 0.0
    metrics.total_decision_points = 0
    metrics.total_drain_event_count = 0
    metrics.total_drained_mu_count = 0
    metrics.mean_drain_count = 0.0
    metrics.total_action_counts = (0, 0, 0)
    metrics.action_fractions = (0.0, 0.0, 0.0)

    captured_callback = None

    def fake_runner(
        *,
        environment,
        agent,
        num_episodes,
        start_episode,
        initial_decision_count,
        on_episode_complete,
    ):
        nonlocal captured_callback

        captured_callback = on_episode_complete

        return [], metrics

    checkpoint_manager = Mock()

    checkpoint_manager_class = Mock(
        return_value=checkpoint_manager
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningCheckpointManager",
        checkpoint_manager_class,
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_environment",
        Mock(return_value=environment),
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_tabular_agent",
        Mock(return_value=agent),
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "run_tabular_q_learning_experiment",
        fake_runner,
    )

    plotter = Mock()

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningMetricsPlotter",
        Mock(return_value=plotter),
    )

    monkeypatch.setattr(
        tabular_q_learning_main.plt,
        "show",
        Mock(),
    )

    tabular_q_learning_main.run(
        plant_settings=Mock(),
        num_episodes=1,
    )

    checkpoint_manager_class.assert_called_once_with(
        tabular_q_learning_main.CHECKPOINT_SETTINGS,
    )

    assert captured_callback is not None

    captured_callback(
        7,
        Mock(),
        metrics,
        123,
    )

    checkpoint_manager.save.assert_called_once_with(
        agent=agent,
        episode_count=7,
        decision_count=123,
        metadata=(
            tabular_q_learning_main.EXPERIMENT_SETTINGS.to_metadata()
        ),
    )

    environment.stop.assert_called_once()
    environment.plant.close_model.assert_called_once()
    environment.plant.quit.assert_called_once()
    environment.tcp_client.close.assert_called_once()

def test_learn_uses_feasible_actions_for_bootstrap(
    monkeypatch,
) -> None:
    q_table = QTable()

    learner = make_learner(
        q_table,
        feasible=True,
    )

    feasible_actions = Mock(
        return_value=(0, 2)
    )

    monkeypatch.setattr(
        ManufacturingProblem,
        "feasible_actions",
        feasible_actions,
    )

    learner.updater.update = Mock(
        return_value=0.0
    )

    state = make_state()

    transition = Mock()
    transition.state = state
    transition.action = 0
    transition.reward = 1.0
    transition.next_state = state
    transition.terminated = False

    learner.learn(transition)

    feasible_actions.assert_called_once_with(
        state
    )

    learner.updater.update.assert_called_once()

    assert (
        learner.updater.update.call_args.kwargs["actions"]
        == (0, 2)
    )


def test_learn_uses_all_actions_without_feasibility_mask(
    monkeypatch,
) -> None:
    q_table = QTable()

    learner = make_learner(
        q_table,
        feasible=False,
    )

    feasible_actions = Mock(
        return_value=(0, 2)
    )

    monkeypatch.setattr(
        ManufacturingProblem,
        "feasible_actions",
        feasible_actions,
    )

    learner.updater.update = Mock(
        return_value=0.0
    )

    state = make_state()

    transition = Mock()
    transition.state = state
    transition.action = 1
    transition.reward = 1.0
    transition.next_state = state
    transition.terminated = False

    learner.learn(transition)

    feasible_actions.assert_not_called()

    learner.updater.update.assert_called_once()

    assert (
        learner.updater.update.call_args.kwargs["actions"]
        == (0, 1, 2)
    )


def test_learner_rejects_non_boolean_feasible() -> None:
    q_table = QTable()
    problem = ManufacturingProblem()

    agent = QLearningAgent(
        q_table=q_table,
        problem=problem,
        settings=QLearningAgentSettings(),
    )

    updater = QLearningUpdater(
        q_table=q_table,
        settings=QLearningSettings(),
    )

    with pytest.raises(
        TypeError,
        match="feasible must be a boolean",
    ):
        TabularQLearner(
            agent=agent,
            updater=updater,
            problem=problem,
            feasible=1,
        )