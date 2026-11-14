from __future__ import annotations

from dataclasses import replace
from unittest.mock import ANY, Mock

import pytest

import tabular_q_learning_main
from config.tcp_settings import TCPSettings
from problem.problem import ManufacturingProblem
from problem.q_learning_agent import QLearningAgentSettings


def test_create_problem_uses_experiment_settings() -> None:
    experiment_settings = (
        tabular_q_learning_main.QLearningExperimentSettings(
            target_throughput=123,
            storage_capacity=7,
            admission_buffer_capacity=42,
            actions=(0, 1, 2),
            learning_rate=0.2,
            discount_factor=0.95,
            epsilon_start=0.9,
            epsilon_min=0.1,
            epsilon_decay=0.8,
            reward_settings=(
                tabular_q_learning_main.REWARD_SETTINGS
            ),
            max_steps=4096
        )
    )

    problem = tabular_q_learning_main.create_problem(
        experiment_settings
    )

    assert problem.target_throughput == 123
    assert problem.storage_capacity == 7
    assert problem.admission_buffer_capacity == 42
    assert problem.actions == (0, 1, 2)


def test_create_environment_constructs_dependency_graph(
    monkeypatch,
) -> None:
    plant_settings = Mock()

    tcp_settings = TCPSettings(
        host="127.0.0.1",
        port=5000,
        receive_timeout=60.0,
    )

    problem = ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        actions=(0, 1, 2),
    )

    fake_plant = Mock()

    plant_factory = Mock(
        return_value=fake_plant
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "PlantSimulation",
        plant_factory,
    )

    fake_tcp_client = Mock()

    tcp_connect = Mock(
        return_value=fake_tcp_client
    )

    monkeypatch.setattr(
        tabular_q_learning_main.TCPClient,
        "connect",
        tcp_connect,
    )

    fake_observation_source = Mock()

    observation_source_factory = Mock(
        return_value=fake_observation_source
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "PlantObservationReader",
        observation_source_factory,
    )

    fake_emptying_event_reader = Mock()

    emptying_event_reader_factory = Mock(
        return_value=fake_emptying_event_reader
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "EmptyingEventReader",
        emptying_event_reader_factory,
    )

    fake_observation_encoder = Mock()

    observation_encoder_factory = Mock(
        return_value=fake_observation_encoder
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "ObservationEncoder",
        observation_encoder_factory,
    )

    fake_reward_function = Mock()

    reward_factory = Mock(
        return_value=fake_reward_function
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningReward",
        reward_factory,
    )

    fake_event_controller_observer = Mock()

    event_controller_observer_factory = Mock(
        return_value=fake_event_controller_observer
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "EventControllerObserver",
        event_controller_observer_factory,
    )

    fake_return_event_reader = Mock()

    return_event_reader_factory = Mock(
        return_value=fake_return_event_reader
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "ReturnEventReader",
        return_event_reader_factory,
    )

    fake_environment = Mock()

    environment_factory = Mock(
        return_value=fake_environment
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "PlantEnvironment",
        environment_factory,
    )

    environment = (
        tabular_q_learning_main.create_environment(
            plant_settings=plant_settings,
            tcp_settings=tcp_settings,
            problem=problem,
        )
    )

    plant_factory.assert_called_once_with(
        settings=plant_settings,
    )

    fake_plant.connect.assert_called_once_with()
    fake_plant.load_model.assert_called_once_with()

    tcp_connect.assert_called_once_with(
        tcp_settings,
        protocol_version="1.0",
    )

    observation_source_factory.assert_called_once_with(
        plant=fake_plant,
    )

    emptying_event_reader_factory.assert_called_once_with(
        plant=fake_plant,
    )

    observation_encoder_factory.assert_called_once()

    reward_factory.assert_called_once_with(
        tabular_q_learning_main.REWARD_SETTINGS,
        storage_capacity=problem.storage_capacity,
    )

    event_controller_observer_factory.assert_called_once_with()

    return_event_reader_factory.assert_called_once_with(
        plant=fake_plant,
    )

    environment_factory.assert_called_once_with(
        plant=fake_plant,
        tcp_client=fake_tcp_client,
        observation_source=fake_observation_source,
        observation_encoder=fake_observation_encoder,
        reward_calculator=fake_reward_function,
        problem=problem,
        event_controller_observer=(
            fake_event_controller_observer
        ),
        emptying_event_reader=(
            fake_emptying_event_reader
        ),
        return_event_reader=(
            fake_return_event_reader
        ),
        settings=(
            tabular_q_learning_main.PlantEnvironmentSettings(
                max_steps=4096,
            )
        ),
    )

    assert environment is fake_environment


def test_create_tabular_agent_builds_direct_q_learning_agent() -> None:
    problem = ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        actions=(0, 1, 2),
    )

    agent = (
        tabular_q_learning_main.create_tabular_agent(
            problem=problem,
        )
    )

    assert isinstance(
        agent,
        tabular_q_learning_main.TabularQLearningAgent,
    )

    assert agent.learner.problem is problem
    assert agent.state_encoder._problem is problem


def test_run_rejects_non_positive_episode_count(
    monkeypatch,
) -> None:
    plant_settings = Mock()

    with pytest.raises(
        ValueError,
        match="num_episodes must be positive",
    ):
        tabular_q_learning_main.run(
            plant_settings=plant_settings,
            num_episodes=0,
        )


def test_run_uses_one_manufacturing_problem_for_dependency_graph(
    monkeypatch,
) -> None:
    plant_settings = Mock()

    tcp_settings = TCPSettings(
        host="127.0.0.1",
        port=5000,
        receive_timeout=60.0,
    )

    problem = Mock()
    environment = Mock()
    agent = Mock()

    create_problem = Mock(
        return_value=problem
    )

    create_environment = Mock(
        return_value=environment
    )

    create_tabular_agent = Mock(
        return_value=agent
    )

    checkpoint_manager = Mock()

    results = [
        Mock(
            episode_number=1,
            outcome=Mock(value="success"),
            decision_points=1,
        )
    ]

    metrics = Mock(
        mean_reward=0.0,
        mean_throughput=1.0,
        total_decision_points=1,
        total_drain_event_count=0,
        total_drained_mu_count=0,
        mean_drain_count=0.0,
        total_action_counts=(1, 0, 0),
        action_fractions=(1.0, 0.0, 0.0),
    )

    run_experiment = Mock(
        return_value=(
            results,
            metrics,
        )
    )

    plotter = Mock()

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_problem",
        create_problem,
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_environment",
        create_environment,
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_tabular_agent",
        create_tabular_agent,
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningCheckpointManager",
        Mock(
            return_value=checkpoint_manager,
        ),
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "run_tabular_q_learning_experiment",
        run_experiment,
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningMetricsPlotter",
        Mock(
            return_value=plotter,
        ),
    )

    monkeypatch.setattr(
        tabular_q_learning_main.plt,
        "show",
        Mock(),
    )

    tabular_q_learning_main.run(
        plant_settings=plant_settings,
        tcp_settings=tcp_settings,
        num_episodes=1,
    )

    create_problem.assert_called_once_with(
        tabular_q_learning_main.EXPERIMENT_SETTINGS
    )

    create_environment.assert_called_once_with(
        plant_settings=plant_settings,
        tcp_settings=tcp_settings,
        problem=problem,
    )

    create_tabular_agent.assert_called_once_with(
        problem=problem,
        experiment_settings=(
            tabular_q_learning_main.EXPERIMENT_SETTINGS
        ),
    )

    run_experiment.assert_called_once_with(
        environment=environment,
        agent=agent,
        num_episodes=1,
        start_episode=1,
        initial_decision_count=0,
        on_episode_complete=ANY,
    )

    environment.stop.assert_called_once_with()
    environment.tcp_client.close.assert_called_once_with()


def test_run_defaults_tcp_settings_when_omitted(
    monkeypatch,
) -> None:
    plant_settings = Mock()
    problem = Mock()
    environment = Mock()
    agent = Mock()

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_problem",
        Mock(
            return_value=problem
        ),
    )

    create_environment = Mock(
        return_value=environment
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_environment",
        create_environment,
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_tabular_agent",
        Mock(
            return_value=agent
        ),
    )

    checkpoint_manager = Mock()

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningCheckpointManager",
        Mock(
            return_value=checkpoint_manager
        ),
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "run_tabular_q_learning_experiment",
        Mock(
            return_value=(
                [],
                Mock(
                    mean_reward=0.0,
                    mean_throughput=0.0,
                    total_decision_points=0,
                    total_drain_event_count=0,
                    total_drained_mu_count=0,
                    mean_drain_count=0.0,
                    total_action_counts=(0, 0, 0),
                    action_fractions=(0.0, 0.0, 0.0),
                ),
            )
        ),
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningMetricsPlotter",
        Mock(
            return_value=Mock()
        ),
    )

    monkeypatch.setattr(
        tabular_q_learning_main.plt,
        "show",
        Mock(),
    )

    tabular_q_learning_main.run(
        plant_settings=plant_settings,
        num_episodes=1,
    )

    create_environment.assert_called_once()

    _, kwargs = create_environment.call_args

    assert kwargs["plant_settings"] is plant_settings

    assert isinstance(
        kwargs["tcp_settings"],
        TCPSettings,
    )

    assert kwargs["problem"] is problem


@pytest.mark.parametrize(
    "feasible",
    [True, False],
)
def test_create_tabular_agent_propagates_feasible_setting(
    monkeypatch,
    feasible: bool,
) -> None:
    problem = ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        actions=(0, 1, 2),
        admission_buffer_capacity=64,
    )

    experiment_settings = replace(
        tabular_q_learning_main.EXPERIMENT_SETTINGS,
        feasible=feasible,
    )

    q_agent = Mock()

    q_agent_factory = Mock(
        return_value=q_agent,
    )

    updater = Mock()

    updater_factory = Mock(
        return_value=updater,
    )

    learner = Mock()

    learner_factory = Mock(
        return_value=learner,
    )

    tabular_agent = Mock()

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningAgent",
        q_agent_factory,
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningUpdater",
        updater_factory,
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "TabularQLearner",
        learner_factory,
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "TabularQLearningAgent",
        Mock(
            return_value=tabular_agent
        ),
    )

    result = (
        tabular_q_learning_main.create_tabular_agent(
            problem=problem,
            experiment_settings=experiment_settings,
        )
    )

    assert result is tabular_agent

    q_agent_factory.assert_called_once()

    settings = (
        q_agent_factory.call_args.kwargs["settings"]
    )

    assert isinstance(
        settings,
        QLearningAgentSettings,
    )

    assert settings.feasible is feasible


def test_run_allows_feasible_change_when_resuming(
    monkeypatch,
) -> None:
    plant_settings = Mock()
    tcp_settings = Mock()

    problem = Mock()
    environment = Mock()
    agent = Mock()

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_problem",
        Mock(
            return_value=problem
        ),
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_environment",
        Mock(
            return_value=environment
        ),
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "create_tabular_agent",
        Mock(
            return_value=agent
        ),
    )

    checkpoint_manager = Mock()

    checkpoint = Mock(
        episode_count=10,
        decision_count=1000,
    )

    checkpoint_manager.restore_from_checkpoint.return_value = (
        checkpoint
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningCheckpointManager",
        Mock(
            return_value=checkpoint_manager,
        ),
    )

    metrics = Mock(
        mean_reward=0.0,
        mean_throughput=0.0,
        total_decision_points=0,
        total_drain_event_count=0,
        total_drained_mu_count=0,
        mean_drain_count=0.0,
        total_action_counts=(0, 0, 0),
        action_fractions=(0.0, 0.0, 0.0),
    )

    experiment_runner = Mock(
        return_value=(
            [],
            metrics,
        ),
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "run_tabular_q_learning_experiment",
        experiment_runner,
    )

    plotter = Mock()

    monkeypatch.setattr(
        tabular_q_learning_main,
        "QLearningMetricsPlotter",
        Mock(
            return_value=plotter,
        ),
    )

    monkeypatch.setattr(
        tabular_q_learning_main.plt,
        "show",
        Mock(),
    )

    monkeypatch.setattr(
        tabular_q_learning_main,
        "EXPERIMENT_SETTINGS",
        replace(
            tabular_q_learning_main.EXPERIMENT_SETTINGS,
            feasible=False,
        ),
    )

    tabular_q_learning_main.run(
        plant_settings=plant_settings,
        tcp_settings=tcp_settings,
        num_episodes=1,
        resume=True,
    )

    checkpoint_manager.restore_from_checkpoint.assert_called_once_with(
        agent=agent,
        experiment_settings=(
            tabular_q_learning_main.EXPERIMENT_SETTINGS
        ),
        allow_feasible_change=True,
    )

    experiment_runner.assert_called_once_with(
        environment=environment,
        agent=agent,
        num_episodes=1,
        start_episode=11,
        initial_decision_count=1000,
        on_episode_complete=ANY,
    )

    environment.stop.assert_called_once_with()
    environment.tcp_client.close.assert_called_once_with()

@pytest.mark.parametrize(
    "feasible",
    [
        True,
        False,
    ],
)
def test_create_tabular_agent_propagates_feasibility_setting(
    feasible: bool,
) -> None:
    experiment_settings = replace(
        tabular_q_learning_main.EXPERIMENT_SETTINGS,
        feasible=feasible,
    )

    problem = ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        admission_buffer_capacity=10,
        actions=(0, 1, 2),
    )

    agent = tabular_q_learning_main.create_tabular_agent(
        problem=problem,
        experiment_settings=experiment_settings,
    )

    assert (
        agent.learner.agent.settings.feasible
        is feasible
    )