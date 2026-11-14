from __future__ import annotations

import random

import numpy as np
import pytest

from checkpoint.checkpoint import (
    CHECKPOINT_VERSION,
    Checkpoint,
)


def _make_checkpoint() -> Checkpoint:
    agent_rng = random.Random(123)

    return Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=7,
        decision_count=143,
        agent_state={
            "selection_count": 347,
            "rng_state": agent_rng.getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata={
            "target_throughput": 1000,
            "storage_capacity": 10,
            "admission_buffer_capacity": 32,
            "actions": (0, 1, 2),
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.998,
        },
    )


def test_checkpoint_has_expected_version() -> None:
    checkpoint = _make_checkpoint()

    assert checkpoint.version == CHECKPOINT_VERSION


def test_checkpoint_preserves_q_table() -> None:
    checkpoint = _make_checkpoint()

    assert checkpoint.q_table == {}


def test_checkpoint_preserves_episode_count() -> None:
    checkpoint = _make_checkpoint()

    assert checkpoint.episode_count == 7


def test_checkpoint_preserves_decision_count() -> None:
    checkpoint = _make_checkpoint()

    assert checkpoint.decision_count == 143


def test_checkpoint_preserves_agent_state() -> None:
    checkpoint = _make_checkpoint()

    assert checkpoint.agent_state["selection_count"] == 347
    assert "rng_state" in checkpoint.agent_state


@pytest.mark.parametrize(
    "field",
    [
        "selection_count",
        "rng_state",
    ],
)
def test_checkpoint_requires_agent_state_fields(
    field: str,
) -> None:
    agent_state = {
        "selection_count": 347,
        "rng_state": random.Random(123).getstate(),
    }

    del agent_state[field]

    with pytest.raises(ValueError):
        Checkpoint(
            version=CHECKPOINT_VERSION,
            q_table={},
            episode_count=0,
            decision_count=0,
            agent_state=agent_state,
            python_random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
            metadata={},
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_count", -1),
        ("decision_count", -1),
    ],
)
def test_checkpoint_rejects_negative_counters(
    field: str,
    value: int,
) -> None:
    kwargs = {
        "version": CHECKPOINT_VERSION,
        "q_table": {},
        "episode_count": 0,
        "decision_count": 0,
        "agent_state": {
            "selection_count": 347,
            "rng_state": random.Random(123).getstate(),
        },
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "metadata": {},
    }

    kwargs[field] = value

    with pytest.raises(ValueError):
        Checkpoint(**kwargs)


def test_checkpoint_rejects_non_positive_version() -> None:
    with pytest.raises(ValueError):
        Checkpoint(
            version=0,
            q_table={},
            episode_count=0,
            decision_count=0,
            agent_state={
                "selection_count": 0,
                "rng_state": random.Random(123).getstate(),
            },
            python_random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
            metadata={},
        )


def test_checkpoint_is_immutable() -> None:
    checkpoint = _make_checkpoint()

    with pytest.raises(AttributeError):
        checkpoint.episode_count = 99


def test_checkpoint_is_dataclass() -> None:
    checkpoint = _make_checkpoint()

    assert checkpoint is not None
    assert hasattr(
        checkpoint,
        "__dataclass_fields__",
    )


def test_checkpoint_preserves_python_random_state() -> None:
    python_state = random.getstate()

    checkpoint = Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=0,
        decision_count=0,
        agent_state={
            "selection_count": 0,
            "rng_state": random.Random(123).getstate(),
        },
        python_random_state=python_state,
        numpy_random_state=np.random.get_state(),
        metadata={},
    )

    assert checkpoint.python_random_state == python_state


def test_checkpoint_preserves_numpy_random_state() -> None:
    numpy_state = np.random.get_state()

    checkpoint = Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=0,
        decision_count=0,
        agent_state={
            "selection_count": 0,
            "rng_state": random.Random(123).getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=numpy_state,
        metadata={},
    )

    assert checkpoint.numpy_random_state[0] == numpy_state[0]
    assert checkpoint.numpy_random_state[2:] == numpy_state[2:]

    assert np.array_equal(
        checkpoint.numpy_random_state[1],
        numpy_state[1],
    )


def test_checkpoint_preserves_metadata() -> None:
    checkpoint = _make_checkpoint()

    assert checkpoint.metadata["target_throughput"] == 1000
    assert checkpoint.metadata["storage_capacity"] == 10
    assert checkpoint.metadata["admission_buffer_capacity"] == 32
    assert checkpoint.metadata["actions"] == (0, 1, 2)