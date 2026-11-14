from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

import tabular_q_learning_evaluate
from agent.tabular_q_learning_agent import (
    TabularQLearningAgent,
)
from problem.q_learning_agent import (
    QLearningAgentSettings,
)

from tabular_q_learning_evaluate import _set_greedy_policy, EvaluationSettings

def _make_fake_agent() -> Mock:
    agent = Mock()
    agent.learner = Mock()
    agent.learner.agent = Mock()
    return agent

def test_set_greedy_policy_disables_exploration() -> None:
    fake_agent = _make_fake_agent()

    fake_learning_agent = Mock()

    fake_agent.learner = Mock()
    fake_agent.learner.agent = fake_learning_agent

    tabular_q_learning_evaluate._set_greedy_policy(
        fake_agent,
        tabular_q_learning_evaluate.EvaluationSettings(
            feasible=False,
        ),
    )

    settings = fake_learning_agent.settings

    assert isinstance(
        settings,
        QLearningAgentSettings,
    )

    assert settings.epsilon_start == 0.0
    assert settings.epsilon_min == 0.0
    assert settings.epsilon_decay == 1.0


def test_evaluate_rejects_non_positive_episode_count(
        monkeypatch
) -> None:
    plant_settings = Mock()
    tcp_settings = Mock()

    monkeypatch.setattr(
        tabular_q_learning_evaluate.plt,
        "show",
        Mock(),
    )

    with pytest.raises(
        ValueError,
        match="num_episodes must be positive",
    ):
        tabular_q_learning_evaluate.evaluate(
            plant_settings=plant_settings,
            tcp_settings=tcp_settings,
            num_episodes=0,
        )


def test_evaluate_rejects_missing_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plant_settings = Mock()
    tcp_settings = Mock()

    checkpoint_path = (
        tmp_path / "missing_checkpoint.pkl"
    )

    environment_factory = Mock()

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "create_environment",
        environment_factory,
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate.plt,
        "show",
        Mock(),
    )

    with pytest.raises(
        FileNotFoundError,
        match="No Q-learning checkpoint exists",
    ):
        tabular_q_learning_evaluate.evaluate(
            plant_settings=plant_settings,
            tcp_settings=tcp_settings,
            num_episodes=1,
            checkpoint_path=checkpoint_path,
        )

    environment_factory.assert_not_called()


def test_evaluate_restores_checkpoint_and_runs_without_learning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plant_settings = Mock()
    tcp_settings = Mock()

    checkpoint_path = (
        tmp_path / "tabular_q_learning.pkl"
    )
    checkpoint_path.touch()

    fake_problem = Mock()
    fake_environment = Mock()
    fake_agent = _make_fake_agent()

    fake_agent.learner.last_q_state = None

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "create_problem",
        Mock(return_value=fake_problem),
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "create_environment",
        Mock(return_value=fake_environment),
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "create_tabular_agent",
        Mock(return_value=fake_agent),
    )

    checkpoint_manager = Mock()
    checkpoint_manager.exists = True

    checkpoint = Mock()
    checkpoint.episode_count = 27
    checkpoint.decision_count = 32400
    checkpoint.q_table = {}

    checkpoint_manager.restore_from_checkpoint.return_value = (
        checkpoint
    )

    checkpoint_manager_factory = Mock(
        return_value=checkpoint_manager
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "QLearningCheckpointManager",
        checkpoint_manager_factory,
    )

    runner = Mock()

    runner.run.return_value = (
        [],
        Mock(),
    )

    runner_factory = Mock(
        return_value=runner
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "QLearningEpisodeRunner",
        runner_factory,
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "_print_evaluation_results",
        Mock(),
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate.plt,
        "show",
        Mock(),
    )

    tabular_q_learning_evaluate.evaluate(
        plant_settings=plant_settings,
        tcp_settings=tcp_settings,
        num_episodes=5,
        checkpoint_path=checkpoint_path,
    )

    checkpoint_manager.restore_from_checkpoint.assert_called_once_with(
        agent=fake_agent,
        experiment_settings=(
            tabular_q_learning_evaluate.EXPERIMENT_SETTINGS
        ),
    )

    runner_factory.assert_called_once_with(
        environment=fake_environment,
        agent=fake_agent,
        learn=False,
    )

    runner.run.assert_called_once_with(
        num_episodes=5,
    )

    fake_environment.stop.assert_called_once_with()

def test_evaluate_cleans_up_environment_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plant_settings = Mock()
    tcp_settings = Mock()

    checkpoint_path = (
        tmp_path / "tabular_q_learning.pkl"
    )
    checkpoint_path.touch()

    fake_problem = Mock()
    fake_environment = Mock()
    fake_agent = _make_fake_agent()

    fake_learning_agent = Mock()

    fake_agent.learner = Mock()
    fake_agent.learner.agent = fake_learning_agent

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "create_problem",
        Mock(return_value=fake_problem),
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "create_environment",
        Mock(return_value=fake_environment),
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "create_tabular_agent",
        Mock(return_value=fake_agent),
    )

    checkpoint_manager = Mock()
    checkpoint_manager.exists = True
    checkpoint_manager.restore_from_checkpoint.return_value = (
        Mock(
            episode_count=27,
            decision_count=32400,
        )
    )

    checkpoint = Mock(
        episode_count=27,
        decision_count=32400,
        q_table={},
    )

    checkpoint_manager.restore_from_checkpoint.return_value = (
        checkpoint
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "QLearningCheckpointManager",
        Mock(
            return_value=checkpoint_manager
        ),
    )

    runner = Mock()
    runner.run.side_effect = RuntimeError(
        "evaluation failure"
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate,
        "QLearningEpisodeRunner",
        Mock(return_value=runner),
    )

    monkeypatch.setattr(
        tabular_q_learning_evaluate.plt,
        "show",
        Mock(),
    )

    with pytest.raises(
        RuntimeError,
        match="evaluation failure",
    ):
        tabular_q_learning_evaluate.evaluate(
            plant_settings=plant_settings,
            tcp_settings=tcp_settings,
            num_episodes=5,
            checkpoint_path=checkpoint_path,
        )

    fake_environment.stop.assert_called_once_with()
    fake_environment.plant.close_model.assert_called_once_with()
    fake_environment.plant.quit.assert_called_once_with()
    fake_environment.tcp_client.close.assert_called_once_with()

def test_evaluation_settings_default_to_infeasible() -> None:
    settings = EvaluationSettings()

    assert settings.feasible is False

@pytest.mark.parametrize(
    "feasible",
    [True, False],
)
def test_evaluation_settings_accept_feasible_setting(
    feasible: bool,
) -> None:
    settings = EvaluationSettings(
        feasible=feasible,
    )

    assert settings.feasible is feasible

@pytest.mark.parametrize(
    "feasible",
    [True, False],
)
def test_set_greedy_policy_applies_evaluation_settings(
    feasible: bool,
) -> None:
    learning_agent = Mock()

    learner = Mock()
    learner.agent = learning_agent

    agent = Mock(
        spec=TabularQLearningAgent,
    )
    agent.learner = learner

    evaluation_settings = EvaluationSettings(
        feasible=feasible,
    )

    _set_greedy_policy(
        agent,
        evaluation_settings,
    )

    assert learning_agent.settings.epsilon_start == 0.0
    assert learning_agent.settings.epsilon_min == 0.0
    assert learning_agent.settings.epsilon_decay == 1.0
    assert learning_agent.settings.feasible is feasible