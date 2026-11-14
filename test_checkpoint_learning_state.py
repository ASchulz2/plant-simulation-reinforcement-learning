from __future__ import annotations

import random

import numpy as np

from checkpoint.checkpoint import (
    CHECKPOINT_VERSION,
    Checkpoint,
)
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


def _make_problem() -> ManufacturingProblem:
    return ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        actions=(0, 1, 2),
        admission_buffer_capacity=32,
    )


def _make_learner() -> TabularQLearner:
    problem = _make_problem()

    table = QTable()

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.998,
        ),
        rng=random.Random(123),
    )

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(
            learning_rate=0.1,
            discount_factor=0.99,
        ),
    )

    return TabularQLearner(
        agent=agent,
        updater=updater,
        problem=problem,
    )


def _checkpoint_from_learner(
    learner: TabularQLearner,
    *,
    episode_count: int = 7,
    decision_count: int = 143,
) -> Checkpoint:
    snapshot = learner.snapshot()

    q_table = snapshot["q_table"]
    agent_state = snapshot["agent_state"]

    assert isinstance(q_table, dict)
    assert isinstance(agent_state, dict)

    return Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table=q_table,
        episode_count=episode_count,
        decision_count=decision_count,
        agent_state=agent_state,
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata={
            "target_throughput": 1000,
            "storage_capacity": 10,
            "admission_buffer_capacity": 32,
            "actions": (0, 1, 2),
        },
    )


def _learner_state_from_checkpoint(
    checkpoint: Checkpoint,
) -> dict[str, object]:
    return {
        "q_table": checkpoint.q_table,
        "agent_state": checkpoint.agent_state,
    }


def test_learner_snapshot_contains_required_state() -> None:
    learner = _make_learner()

    snapshot = learner.snapshot()

    assert set(snapshot) == {
        "q_table",
        "agent_state",
    }

    assert isinstance(snapshot["q_table"], dict)
    assert isinstance(snapshot["agent_state"], dict)


def test_learner_snapshot_can_be_embedded_in_checkpoint() -> None:
    learner = _make_learner()

    checkpoint = _checkpoint_from_learner(learner)

    assert checkpoint.q_table == {}

    assert (
        checkpoint.agent_state["selection_count"]
        == 0
    )

    assert checkpoint.episode_count == 7
    assert checkpoint.decision_count == 143


def test_learner_restore_preserves_agent_state() -> None:
    first = _make_learner()

    # Consume actions to advance the private agent state.
    # The exact state is then captured through snapshot().
    first.agent._selection_count = 25

    snapshot = first.snapshot()

    second = _make_learner()
    second.restore(snapshot)

    assert second.snapshot()["agent_state"] == (
        snapshot["agent_state"]
    )


def test_learner_restore_replaces_existing_q_table() -> None:
    first = _make_learner()

    target_snapshot = first.snapshot()

    second = _make_learner()

    # Introduce different state into the destination table.
    second.agent.q_table._values = {
        ("placeholder", 0): 999.0,
    }

    second.restore(target_snapshot)

    assert second.snapshot()["q_table"] == (
        target_snapshot["q_table"]
    )


def test_learner_checkpoint_roundtrip_preserves_learning_state() -> None:
    first = _make_learner()

    first.agent._selection_count = 17

    checkpoint = _checkpoint_from_learner(
        first,
        episode_count=11,
        decision_count=219,
    )

    second = _make_learner()

    second.restore(
        _learner_state_from_checkpoint(
            checkpoint
        )
    )

    restored_snapshot = second.snapshot()

    assert restored_snapshot["q_table"] == (
        checkpoint.q_table
    )

    assert restored_snapshot["agent_state"] == (
        checkpoint.agent_state
    )