from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from checkpoint.checkpoint import (
    CHECKPOINT_VERSION,
    Checkpoint,
)
from checkpoint.checkpoint_store import CheckpointStore
from checkpoint.q_learning_checkpoint import (
    QLearningCheckpointManager,
    QLearningCheckpointSettings,
    QLearningExperimentSettings,
    QLearningRewardSettings,
)
from dataclasses import replace

def _make_valid_checkpoint(
    *,
    metadata: dict[str, object],
) -> Checkpoint:
    return Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=0,
        decision_count=0,
        agent_state={
            "selection_count": 0,
            "rng_state": random.Random().getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata=metadata,
    )


def _experiment_settings(
    *,
    reward_settings: QLearningRewardSettings | None = None,
    feasible: bool = True,
) -> QLearningExperimentSettings:
    return QLearningExperimentSettings(
        target_throughput=1000,
        storage_capacity=10,
        admission_buffer_capacity=64,
        actions=(0, 1, 2),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.996,
        reward_settings=(
            reward_settings
            if reward_settings is not None
            else QLearningRewardSettings(
                throughput_weight=5.0,
                storage_reward=1.0,
                time_penalty=0.01,
                return_waiting_time_penalty=1.0,
                return_waiting_time_reference=21.0,
                success_bonus=10.0,
                failure_penalty=10.0,
                premature_emptying_penalty=10.0,
                return_action_penalty=1.0,
            )
        ),
        feasible=feasible,
        max_steps=4096
    )


def _checkpoint(
    *,
    metadata: dict[str, object] | None = None,
    episode_count: int = 20,
    decision_count: int = 8000,
) -> Checkpoint:
    return Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={
            ("GER",): {
                0: 1.5,
                1: 2.5,
                2: 3.5,
            }
        },
        episode_count=episode_count,
        decision_count=decision_count,
        agent_state={
            "selection_count": 42,
            "rng_state": random.Random(12345).getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata=(
            metadata
            if metadata is not None
            else {}
        ),
    )


def test_checkpoint_settings_store_path() -> None:
    settings = QLearningCheckpointSettings(
        path=Path("checkpoint.pkl")
    )

    assert settings.path == Path("checkpoint.pkl")


def test_checkpoint_settings_reject_non_path() -> None:
    with pytest.raises(
        TypeError,
        match="path must be a pathlib.Path",
    ):
        QLearningCheckpointSettings(
            path="checkpoint.pkl",  # type: ignore[arg-type]
        )


def test_checkpoint_settings_reject_non_positive_version() -> None:
    with pytest.raises(
        ValueError,
        match="expected_version must be positive",
    ):
        QLearningCheckpointSettings(
            path=Path("checkpoint.pkl"),
            expected_version=0,
        )


def test_checkpoint_manager_reports_missing_checkpoint(
    tmp_path: Path,
) -> None:
    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=tmp_path / "checkpoint.pkl",
        )
    )

    assert manager.exists is False


def test_checkpoint_manager_reports_existing_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pkl"

    checkpoint = _checkpoint()

    CheckpointStore(path).save(checkpoint)

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(path=path)
    )

    assert manager.exists is True


def test_checkpoint_manager_save_persists_learning_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pkl"

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(path=path)
    )

    agent = MagicMock()

    q_table = {
        ("GER",): {
            0: 1.5,
            1: 2.5,
            2: 3.5,
        }
    }

    agent_state = {
        "selection_count": 42,
        "rng_state": ("state",),
    }

    agent.learner.snapshot.return_value = {
        "q_table": q_table,
        "agent_state": agent_state,
    }

    random.seed(12345)
    np.random.seed(54321)

    metadata = {
        "target_throughput": 1000,
        "storage_capacity": 10,
        "actions": (0, 1, 2),
    }

    manager.save(
        agent=agent,
        episode_count=20,
        decision_count=8000,
        metadata=metadata,
    )

    assert manager.exists is True

    stored = CheckpointStore(path).load()

    assert stored.version == CHECKPOINT_VERSION
    assert stored.q_table == q_table
    assert stored.agent_state == agent_state
    assert stored.episode_count == 20
    assert stored.decision_count == 8000
    assert stored.metadata == metadata

    agent.learner.snapshot.assert_called_once_with()


def test_checkpoint_manager_load_returns_saved_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pkl"

    expected = _checkpoint(
        metadata={
            "storage_capacity": 10,
        },
        episode_count=7,
        decision_count=2800,
    )

    CheckpointStore(path).save(expected)

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(path=path)
    )

    loaded = manager.load()

    assert loaded.version == expected.version
    assert loaded.q_table == expected.q_table
    assert loaded.episode_count == expected.episode_count
    assert loaded.decision_count == expected.decision_count
    assert loaded.agent_state == expected.agent_state
    assert loaded.python_random_state == (
        expected.python_random_state
    )
    assert loaded.metadata == expected.metadata

    loaded_numpy = loaded.numpy_random_state
    expected_numpy = expected.numpy_random_state

    assert loaded_numpy[0] == expected_numpy[0]
    assert np.array_equal(
        loaded_numpy[1],
        expected_numpy[1],
    )
    assert loaded_numpy[2:] == expected_numpy[2:]


def test_checkpoint_manager_rejects_version_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pkl"

    settings = _experiment_settings()

    checkpoint = Checkpoint(
        version=CHECKPOINT_VERSION + 1,
        q_table={},
        episode_count=1,
        decision_count=400,
        agent_state={
            "selection_count": 1,
            "rng_state": random.Random(12345).getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata=settings.to_metadata(),
    )

    CheckpointStore(path).save(checkpoint)

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=path,
            expected_version=CHECKPOINT_VERSION,
        )
    )

    with pytest.raises(
        ValueError,
        match="Checkpoint version mismatch",
    ):
        manager.load()


def test_checkpoint_manager_restores_learning_state_and_rng(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pkl"

    settings = _experiment_settings()

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(path=path)
    )

    agent = MagicMock()

    q_table = {
        ("GER",): {
            0: 1.5,
            1: 2.5,
            2: 3.5,
        }
    }

    agent_state = {
        "selection_count": 42,
        "rng_state": ("state",),
    }

    python_rng = random.Random(12345)
    numpy_rng = np.random.RandomState(54321)

    checkpoint = Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table=q_table,
        episode_count=20,
        decision_count=8000,
        agent_state=agent_state,
        python_random_state=python_rng.getstate(),
        numpy_random_state=numpy_rng.get_state(),
        metadata=settings.to_metadata(),
    )

    CheckpointStore(path).save(checkpoint)

    expected_python_value = python_rng.random()
    expected_numpy_value = numpy_rng.rand()

    random.seed(99999)
    np.random.seed(99999)

    manager.restore_from_checkpoint(
        agent=agent,
        experiment_settings=settings,
    )

    assert random.random() == expected_python_value
    assert np.random.rand() == expected_numpy_value

    agent.learner.restore.assert_called_once_with(
        {
            "q_table": q_table,
            "agent_state": agent_state,
        }
    )


def test_experiment_settings_store_configuration() -> None:
    settings = _experiment_settings()

    assert settings.target_throughput == 1000
    assert settings.storage_capacity == 10
    assert settings.admission_buffer_capacity == 64
    assert settings.actions == (0, 1, 2)
    assert settings.learning_rate == 0.1
    assert settings.discount_factor == 0.99
    assert settings.epsilon_start == 1.0
    assert settings.epsilon_min == 0.05
    assert settings.epsilon_decay == 0.996


def test_experiment_settings_create_checkpoint_metadata() -> None:
    settings = _experiment_settings()

    assert settings.to_metadata() == {
        "target_throughput": 1000,
        "storage_capacity": 10,
        "admission_buffer_capacity": 64,
        "actions": (0, 1, 2),
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.996,
        "feasible": True,
        "reward_throughput_weight": 5.0,
        "reward_storage_reward": 1.0,
        "reward_time_penalty": 0.01,
        "reward_return_waiting_time_penalty": 1.0,
        "reward_return_waiting_time_reference": 21.0,
        "reward_success_bonus": 10.0,
        "reward_failure_penalty": 10.0,
        "reward_premature_emptying_penalty": 10.0,
        "reward_return_action_penalty": 1.0,
        "reward_feasibility_reward": 1.0,
        "max_steps": 4096
    }


def test_checkpoint_manager_accepts_compatible_experiment_settings(
    tmp_path: Path,
) -> None:
    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=tmp_path / "checkpoint.pkl",
        )
    )

    settings = _experiment_settings()

    checkpoint = _checkpoint(
        metadata=settings.to_metadata()
    )

    manager.validate_compatibility(
        checkpoint=checkpoint,
        experiment_settings=settings,
    )


def test_checkpoint_manager_rejects_incompatible_storage_capacity(
    tmp_path: Path,
) -> None:
    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=tmp_path / "checkpoint.pkl",
        )
    )

    checkpoint_settings = _experiment_settings()

    current_settings = QLearningExperimentSettings(
        target_throughput=1000,
        storage_capacity=20,
        admission_buffer_capacity=64,
        actions=(0, 1, 2),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.996,
        reward_settings=QLearningRewardSettings(),
        max_steps=4096
    )

    checkpoint = _checkpoint(
        metadata=checkpoint_settings.to_metadata()
    )

    with pytest.raises(
        ValueError,
        match="storage_capacity",
    ):
        manager.validate_compatibility(
            checkpoint=checkpoint,
            experiment_settings=current_settings,
        )


def test_checkpoint_manager_rejects_missing_metadata_field(
    tmp_path: Path,
) -> None:
    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=tmp_path / "checkpoint.pkl",
        )
    )

    settings = _experiment_settings()

    metadata = settings.to_metadata()
    del metadata["epsilon_decay"]

    checkpoint = _checkpoint(
        metadata=metadata
    )

    with pytest.raises(
        ValueError,
        match="epsilon_decay",
    ):
        manager.validate_compatibility(
            checkpoint=checkpoint,
            experiment_settings=settings,
        )


def test_experiment_settings_create_metadata() -> None:
    settings = _experiment_settings()

    metadata = settings.to_metadata()

    assert metadata["target_throughput"] == 1000
    assert metadata["storage_capacity"] == 10
    assert metadata["learning_rate"] == 0.1
    assert metadata["reward_return_action_penalty"] == 1.0


def test_incompatible_reward_settings_are_rejected() -> None:
    checkpoint_settings = _experiment_settings()

    checkpoint = _checkpoint(
        metadata=checkpoint_settings.to_metadata()
    )

    incompatible = _experiment_settings(
        reward_settings=QLearningRewardSettings(
            throughput_weight=5.0,
            storage_reward=1.0,
            time_penalty=0.01,
            return_waiting_time_penalty=1.0,
            return_waiting_time_reference=21.0,
            success_bonus=10.0,
            failure_penalty=10.0,
            premature_emptying_penalty=10.0,
            return_action_penalty=2.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="reward_return_action_penalty",
    ):
        QLearningCheckpointManager(
            QLearningCheckpointSettings(
                path=Path("checkpoint.pkl")
            )
        ).validate_compatibility(
            checkpoint=checkpoint,
            experiment_settings=incompatible,
        )


def test_experiment_settings_default_to_feasible() -> None:
    settings = _experiment_settings()

    assert settings.feasible is True


def test_experiment_settings_support_infeasible_action_selection() -> None:
    settings = _experiment_settings(
        feasible=False,
    )

    assert settings.feasible is False


def test_experiment_settings_metadata_contains_feasible() -> None:
    settings = _experiment_settings(
        feasible=False,
    )

    metadata = settings.to_metadata()

    assert metadata["feasible"] is False


def test_checkpoint_compatibility_rejects_feasible_setting_mismatch() -> None:
    checkpoint_settings = _experiment_settings(
        feasible=False,
    )

    current_settings = _experiment_settings(
        feasible=True,
    )

    checkpoint = Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=0,
        decision_count=0,
        agent_state={
            "selection_count": 0,
            "rng_state": random.Random().getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata=checkpoint_settings.to_metadata(),
    )

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=Path("checkpoint.pkl"),
        )
    )

    with pytest.raises(
        ValueError,
        match="'feasible' differs",
    ):
        manager.validate_compatibility(
            checkpoint=checkpoint,
            experiment_settings=current_settings,
        )


def test_checkpoint_manager_accepts_legacy_checkpoint_without_feasible_metadata() -> None:
    checkpoint_settings = _experiment_settings(
        feasible=True,
    )

    current_settings = _experiment_settings(
        feasible=True,
    )

    metadata = checkpoint_settings.to_metadata()
    metadata.pop("feasible")

    checkpoint = Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=0,
        decision_count=0,
        agent_state={
            "selection_count": 0,
            "rng_state": random.Random().getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata=metadata,
    )

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=Path("checkpoint.pkl"),
        )
    )

    manager.validate_compatibility(
        checkpoint=checkpoint,
        experiment_settings=current_settings,
    )


def test_legacy_checkpoint_is_incompatible_with_feasible_false() -> None:
    checkpoint_settings = _experiment_settings(
        feasible=True,
    )

    current_settings = _experiment_settings(
        feasible=False,
    )

    metadata = checkpoint_settings.to_metadata()
    metadata.pop("feasible")

    checkpoint = Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=0,
        decision_count=0,
        agent_state={
            "selection_count": 0,
            "rng_state": random.Random().getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata=metadata,
    )

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=Path("checkpoint.pkl"),
        )
    )

    with pytest.raises(
        ValueError,
        match="'feasible' differs",
    ):
        manager.validate_compatibility(
            checkpoint=checkpoint,
            experiment_settings=current_settings,
        )


def test_checkpoint_compatibility_rejects_feasible_change_by_default() -> None:
    checkpoint_settings = _experiment_settings(
        feasible=True,
    )

    current_settings = _experiment_settings(
        feasible=False,
    )

    checkpoint = _make_valid_checkpoint(
        metadata=checkpoint_settings.to_metadata(),
    )

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=Path("checkpoint.pkl"),
        )
    )

    with pytest.raises(
        ValueError,
        match="'feasible' differs",
    ):
        manager.validate_compatibility(
            checkpoint=checkpoint,
            experiment_settings=current_settings,
        )


def test_checkpoint_compatibility_allows_feasible_change_for_resume() -> None:
    checkpoint_settings = _experiment_settings(
        feasible=True,
    )

    current_settings = _experiment_settings(
        feasible=False,
    )

    checkpoint = _make_valid_checkpoint(
        metadata=checkpoint_settings.to_metadata(),
    )

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=Path("checkpoint.pkl"),
        )
    )

    manager.validate_compatibility(
        checkpoint=checkpoint,
        experiment_settings=current_settings,
        allow_feasible_change=True,
    )


def test_legacy_checkpoint_without_feasible_allows_resume_with_feasible_false() -> None:
    checkpoint_settings = _experiment_settings(
        feasible=True,
    )

    current_settings = _experiment_settings(
        feasible=False,
    )

    metadata = checkpoint_settings.to_metadata()
    metadata.pop("feasible")

    checkpoint = _make_valid_checkpoint(
        metadata=metadata,
    )

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=Path("checkpoint.pkl"),
        )
    )

    manager.validate_compatibility(
        checkpoint=checkpoint,
        experiment_settings=current_settings,
        allow_feasible_change=True,
    )


def test_checkpoint_metadata_includes_feasibility_reward() -> None:
    settings = QLearningExperimentSettings(
        target_throughput=1000,
        storage_capacity=10,
        admission_buffer_capacity=2048,
        actions=(0, 1, 2),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        reward_settings=QLearningRewardSettings(
            throughput_weight=0.0,
            storage_reward=0.0,
            time_penalty=0.0,
            return_waiting_time_penalty=0.0,
            return_waiting_time_reference=21.0,
            success_bonus=1000.0,
            failure_penalty=1000.0,
            premature_emptying_penalty=0.0,
            return_action_penalty=0.0,
            feasibility_reward=1.0,
        ),
        feasible=False,
        max_steps=4096
    )

    metadata = settings.to_metadata()

    assert metadata["reward_feasibility_reward"] == 1.0


def test_checkpoint_rejects_changed_feasibility_reward(
    tmp_path: Path,
) -> None:
    settings = QLearningExperimentSettings(
        target_throughput=1000,
        storage_capacity=10,
        admission_buffer_capacity=2048,
        actions=(0, 1, 2),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        reward_settings=QLearningRewardSettings(
            throughput_weight=0.0,
            storage_reward=0.0,
            time_penalty=0.0,
            return_waiting_time_penalty=0.0,
            return_waiting_time_reference=21.0,
            success_bonus=1000.0,
            failure_penalty=1000.0,
            premature_emptying_penalty=0.0,
            return_action_penalty=0.0,
            feasibility_reward=1.0,
        ),
        feasible=False,
        max_steps=4096
    )

    metadata = settings.to_metadata()
    metadata["reward_feasibility_reward"] = 0.0

    checkpoint = Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=1,
        decision_count=1,
        agent_state={
            "selection_count": 0,
            "rng_state": random.Random().getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata=metadata,
    )

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=tmp_path / "checkpoint.pkl",
        )
    )

    with pytest.raises(
        ValueError,
        match="reward_feasibility_reward",
    ):
        manager.validate_compatibility(
            checkpoint=checkpoint,
            experiment_settings=settings,
        )

def test_validate_compatibility_rejects_max_steps_change(
    tmp_path: Path,
) -> None:
    experiment_settings = _experiment_settings()

    checkpoint = Checkpoint(
        version=CHECKPOINT_VERSION,
        q_table={},
        episode_count=0,
        decision_count=0,
        agent_state={
            "selection_count": 0,
            "rng_state": random.getstate(),
        },
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        metadata=experiment_settings.to_metadata(),
    )

    manager = QLearningCheckpointManager(
        QLearningCheckpointSettings(
            path=tmp_path / "checkpoint.pkl",
        )
    )

    incompatible_settings = replace(
        experiment_settings,
        max_steps=8192,
    )

    with pytest.raises(
        ValueError,
        match="max_steps.*differs",
    ):
        manager.validate_compatibility(
            checkpoint=checkpoint,
            experiment_settings=incompatible_settings,
        )