from __future__ import annotations

import random

import numpy as np
import pytest

from checkpoint.checkpoint import (
    CHECKPOINT_VERSION,
    Checkpoint,
)
from checkpoint.checkpoint_store import (
    CheckpointStore,
)
import pickle


def _make_checkpoint(
    *,
    episode_count: int = 7,
    decision_count: int = 143,
) -> Checkpoint:
    return Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=episode_count,
        decision_count=decision_count,
        agent_state={
            "selection_count": 347,
            "rng_state": random.Random(123).getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata={
            "target_throughput": 1000,
            "storage_capacity": 10,
            "admission_buffer_capacity": 32,
            "actions": (0, 1, 2),
        },
    )


def test_store_does_not_report_nonexistent_checkpoint(
    tmp_path,
) -> None:
    store = CheckpointStore(
        tmp_path / "checkpoint.pkl"
    )

    assert not store.exists()


def test_store_saves_checkpoint(
    tmp_path,
) -> None:
    path = tmp_path / "checkpoint.pkl"
    store = CheckpointStore(path)

    checkpoint = _make_checkpoint()

    store.save(checkpoint)

    assert path.exists()
    assert store.exists()


def test_store_loads_checkpoint(
    tmp_path,
) -> None:
    path = tmp_path / "checkpoint.pkl"
    store = CheckpointStore(path)

    original = _make_checkpoint()

    store.save(original)
    restored = store.load()

    assert restored.version == original.version
    assert restored.q_table == original.q_table
    assert restored.episode_count == original.episode_count
    assert restored.decision_count == original.decision_count
    assert restored.agent_state == original.agent_state
    assert (
        restored.python_random_state
        == original.python_random_state
    )

    assert restored.numpy_random_state[0] == (
        original.numpy_random_state[0]
    )
    assert np.array_equal(
        restored.numpy_random_state[1],
        original.numpy_random_state[1],
    )

    assert restored.numpy_random_state[2:] == (
        original.numpy_random_state[2:]
    )

    assert restored.metadata == original.metadata


def test_store_load_requires_existing_file(
    tmp_path,
) -> None:
    store = CheckpointStore(
        tmp_path / "missing.pkl"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint not found",
    ):
        store.load()


def test_store_replaces_existing_checkpoint(
    tmp_path,
) -> None:
    path = tmp_path / "checkpoint.pkl"
    store = CheckpointStore(path)

    first = _make_checkpoint(
        episode_count=7,
        decision_count=143,
    )
    second = _make_checkpoint(
        episode_count=12,
        decision_count=251,
    )

    store.save(first)
    store.save(second)

    restored = store.load()

    assert restored.episode_count == 12
    assert restored.decision_count == 251


def test_store_creates_parent_directory(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "nested"
        / "checkpoint"
        / "checkpoint.pkl"
    )

    store = CheckpointStore(path)

    store.save(_make_checkpoint())

    assert path.exists()


def test_store_does_not_leave_temporary_file(
    tmp_path,
) -> None:
    path = tmp_path / "checkpoint.pkl"
    temporary_path = tmp_path / "checkpoint.pkl.tmp"

    store = CheckpointStore(path)

    store.save(_make_checkpoint())

    assert path.exists()
    assert not temporary_path.exists()


def test_store_rejects_invalid_serialized_payload(
    tmp_path,
) -> None:
    path = tmp_path / "checkpoint.pkl"

    with path.open("wb") as handle:
        pickle.dump(
            {"not": "a checkpoint"},
            handle,
        )

    store = CheckpointStore(path)

    with pytest.raises(
        TypeError,
        match="does not contain a Checkpoint",
    ):
        store.load()