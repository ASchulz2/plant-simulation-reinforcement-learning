from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from agent.tabular_q_learning_agent import (
    TabularQLearningAgent,
)
from checkpoint.q_learning_checkpoint import (
    QLearningCheckpointManager,
    QLearningCheckpointSettings,
    QLearningExperimentSettings,
)
from config.settings import PlantSimulationSettings
from config.tcp_settings import TCPSettings
from environment.emptying_event_reader import (
    EmptyingEventReader,
)
from environment.event_controller_observer import (
    EventControllerObserver,
)
from environment.plant_environment import (
    PlantEnvironment,
    PlantEnvironmentSettings,
)
from environment.plant_observation_reader import (
    PlantObservationReader,
)
from environment.q_learning_episode_runner import (
    run_tabular_q_learning_experiment,
)
from environment.return_event_reader import (
    ReturnEventReader,
)
from problem.observation_encoder import (
    ObservationEncoder,
    ObservationEncoderSettings,
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
from problem.reward import (
    QLearningReward,
    QLearningRewardSettings,
)
from problem.tabular_q_learning import (
    TabularQLearner,
)
from problem.tabular_state_encoder import (
    TabularStateEncoder,
)
from simulation.plant_simulation import (
    PlantSimulation,
)
from tcp.client import TCPClient
from visualization.q_learning_metrics_plotter import (
    QLearningMetricsPlotter,
)


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "checkpoint"
    / "tabular_q_learning_pure.pkl"
)


REWARD_SETTINGS = QLearningRewardSettings(
    throughput_weight=0.0,
    storage_reward=0.0,
    time_penalty=0.0,
    return_waiting_time_penalty=0.0,
    return_waiting_time_reference=21.0,
    success_bonus=10.0,
    failure_penalty=10.0,
    premature_emptying_penalty=0.0,
    return_action_penalty=0.0,
    feasibility_reward=1.
)

EXPERIMENT_SETTINGS = QLearningExperimentSettings(
    target_throughput=1000,
    storage_capacity=10,
    admission_buffer_capacity=4096,
    max_steps=4096,
    actions=(0, 1, 2),
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.998,
    reward_settings=REWARD_SETTINGS,
    feasible = True, # feasible = True removes invalid decisions; it does not remove the learning of preference among valid decisions.
)


CHECKPOINT_SETTINGS = QLearningCheckpointSettings(
    path=CHECKPOINT_PATH,
)


def create_environment(
    plant_settings: PlantSimulationSettings,
    tcp_settings: TCPSettings,
    problem: ManufacturingProblem,
) -> PlantEnvironment:
    """
    Construct the complete PlantEnvironment dependency graph.

    Configuration and problem definition are supplied from outside
    this function so that the environment shares the same problem
    instance as the learning agent.
    """

    plant = PlantSimulation(
        settings=plant_settings,
    )

    plant.connect()
    plant.load_model()

    tcp_client = TCPClient.connect(
        tcp_settings,
        protocol_version="1.0",
    )

    observation_source = PlantObservationReader(
        plant=plant,
    )

    emptying_event_reader = EmptyingEventReader(
        plant=plant,
    )

    observation_encoder = ObservationEncoder(
        ObservationEncoderSettings(
            storage_a_normalization_capacity=10,
            storage_b_normalization_capacity=10,
            return_buffer_capacity=10,
            target_throughput=1000,
        )
    )

    reward_function = QLearningReward(
        REWARD_SETTINGS,
        storage_capacity=problem.storage_capacity,
    )

    event_controller_observer = (
        EventControllerObserver()
    )

    return_event_reader = ReturnEventReader(
        plant=plant,
    )

    return PlantEnvironment(
        plant=plant,
        tcp_client=tcp_client,
        observation_source=observation_source,
        observation_encoder=observation_encoder,
        reward_calculator=reward_function,
        problem=problem,
        event_controller_observer=(
            event_controller_observer
        ),
        emptying_event_reader=(
            emptying_event_reader
        ),
        return_event_reader=(
            return_event_reader
        ),
        settings=PlantEnvironmentSettings(
            max_steps=EXPERIMENT_SETTINGS.max_steps,
        ),
    )


def create_tabular_agent(
    problem: ManufacturingProblem,
    experiment_settings: QLearningExperimentSettings = (
        EXPERIMENT_SETTINGS
    ),
) -> TabularQLearningAgent:
    """
    Construct the direct tabular Q-learning agent.

    The supplied ManufacturingProblem instance is shared with
    the environment and the learner.
    """

    table = QTable()

    q_agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=(
                experiment_settings.epsilon_start
            ),
            epsilon_min=(
                experiment_settings.epsilon_min
            ),
            epsilon_decay=(
                experiment_settings.epsilon_decay
            ),
            feasible=(
                experiment_settings.feasible
            )
        ),
    )

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(
            learning_rate=(
                experiment_settings.learning_rate
            ),
            discount_factor=(
                experiment_settings.discount_factor
            ),
        ),
    )

    learner = TabularQLearner(
        agent=q_agent,
        updater=updater,
        problem=problem,
    )

    return TabularQLearningAgent(
        state_encoder=TabularStateEncoder(
            problem=problem,
        ),
        learner=learner,
    )


def create_problem(
    experiment_settings: QLearningExperimentSettings,
) -> ManufacturingProblem:
    """Construct the manufacturing problem from experiment settings."""

    return ManufacturingProblem(
        target_throughput=(
            experiment_settings.target_throughput
        ),
        storage_capacity=(
            experiment_settings.storage_capacity
        ),
        actions=experiment_settings.actions,
        admission_buffer_capacity=(
            experiment_settings.admission_buffer_capacity
        ),
    )


def run(
    *,
    plant_settings: PlantSimulationSettings,
    tcp_settings: TCPSettings | None = None,
    num_episodes: int = 10,
    resume: bool = False,
) -> None:
    """
    Construct and run the real tabular Q-learning experiment.

    Parameters
    ----------
    plant_settings:
        Plant Simulation configuration.

    tcp_settings:
        TCP configuration. Defaults to TCPSettings().

    num_episodes:
        Number of episodes to execute in this invocation.

    resume:
        Whether to restore the most recent checkpoint and continue
        training from it.
    """

    if num_episodes <= 0:
        raise ValueError(
            "num_episodes must be positive."
        )

    if tcp_settings is None:
        tcp_settings = TCPSettings()

    problem = create_problem(
        EXPERIMENT_SETTINGS
    )

    checkpoint_manager = (
        QLearningCheckpointManager(
            CHECKPOINT_SETTINGS
        )
    )

    environment = create_environment(
        plant_settings=plant_settings,
        tcp_settings=tcp_settings,
        problem=problem,
    )

    agent = create_tabular_agent(
        problem=problem,
        experiment_settings=EXPERIMENT_SETTINGS,
    )

    start_episode = 1
    initial_decision_count = 0

    if resume:
        checkpoint = (
            checkpoint_manager.restore_from_checkpoint(
                agent=agent,
                experiment_settings=EXPERIMENT_SETTINGS,
                allow_feasible_change=True,
            )
        )

        start_episode = (
            checkpoint.episode_count + 1
        )

        initial_decision_count = (
            checkpoint.decision_count
        )

        print(
            "Resuming tabular Q-learning from "
            f"episode {checkpoint.episode_count} "
            f"({checkpoint.decision_count} "
            "decision points)."
        )

    def on_episode_complete(
        episode_number: int,
        _result,
        _metrics,
        decision_count: int,
    ) -> None:
        checkpoint_manager.save(
            agent=agent,
            episode_count=episode_number,
            decision_count=decision_count,
            metadata=EXPERIMENT_SETTINGS.to_metadata(),
        )

        print(
            f"Checkpoint saved after episode "
            f"{episode_number} "
            f"({decision_count} decision points)."
        )

    try:
        results, metrics = (
            run_tabular_q_learning_experiment(
                environment=environment,
                agent=agent,
                num_episodes=num_episodes,
                start_episode=start_episode,
                initial_decision_count=(
                    initial_decision_count
                ),
                on_episode_complete=(
                    on_episode_complete
                ),
            )
        )

        plotter = QLearningMetricsPlotter()

        plotter.plot_total_reward(metrics)
        plotter.plot_throughput(metrics)
        plotter.plot_decision_points(metrics)
        plotter.plot_mean_drain_count(metrics)
        plotter.plot_simulation_time(metrics)

        plt.show()

        print(
            "Episodes completed:",
            len(results),
        )

        print()
        print("=" * 70)
        print(
            "TABULAR Q-LEARNING EXPERIMENT FINISHED"
        )
        print("=" * 70)

        for result in results:
            print(
                f"Episode {result.episode_number}: "
                f"{result.outcome.value}, "
                f"decision points="
                f"{result.decision_points}"
            )

        print()
        print("Experiment metrics:")

        print(
            "Mean reward:",
            metrics.mean_reward,
        )

        print(
            "Mean throughput:",
            metrics.mean_throughput,
        )

        print(
            "Total decision points:",
            metrics.total_decision_points,
        )

        print(
            "Total drain events:",
            metrics.total_drain_event_count,
        )

        print(
            "Total drained MUs:",
            metrics.total_drained_mu_count,
        )

        print(
            "Mean drain count:",
            metrics.mean_drain_count,
        )

        print()
        print("Action usage:")

        action_counts = metrics.total_action_counts
        action_fractions = metrics.action_fractions

        print(
            f"StorageA (0): {action_counts[0]} "
            f"({action_fractions[0]:.2%})"
        )

        print(
            f"StorageB (1): {action_counts[1]} "
            f"({action_fractions[1]:.2%})"
        )

        print(
            f"Return   (2): {action_counts[2]} "
            f"({action_fractions[2]:.2%})"
        )

        print()
        print(
            "Distinct visited states:",
            agent.learner.state_registry.count(),
        )

        print(
            "Distinct visited state-action pairs:",
            agent.learner.state_action_registry.count(),
        )

    finally:
        environment.stop()

        try:
            environment.plant.close_model()
        except Exception as exc:
            print(
                "Warning: failed to close Plant Simulation model:",
                exc,
            )

        try:
            environment.plant.quit()
        except Exception as exc:
            print(
                "Warning: failed to quit Plant Simulation:",
                exc,
            )

        environment.tcp_client.close()


if __name__ == "__main__":
    raise SystemExit(
        "Import run() and provide PlantSimulationSettings "
        "from your application configuration."
    )