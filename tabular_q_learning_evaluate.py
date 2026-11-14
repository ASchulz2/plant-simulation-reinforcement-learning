from __future__ import annotations

from pathlib import Path

from agent.tabular_q_learning_agent import (
    TabularQLearningAgent,
)
from checkpoint.q_learning_checkpoint import (
    QLearningCheckpointManager,
    QLearningCheckpointSettings,
)
from config.settings import PlantSimulationSettings
from config.tcp_settings import TCPSettings
from environment.q_learning_episode_runner import (
    QLearningEpisodeRunner,
)
from environment.q_learning_metrics import (
    QLearningExperimentMetrics,
)
from environment.episode import (
    EpisodeOutcome,
)
from problem.q_learning_agent import (
    QLearningAgentSettings,
)

from tabular_q_learning_main import (
    CHECKPOINT_PATH,
    EXPERIMENT_SETTINGS,
    create_environment,
    create_problem,
    create_tabular_agent,
)

from visualization.q_table_plotter import QTablePlotter

from dataclasses import dataclass

import matplotlib.pyplot as plt

@dataclass(frozen=True)
class EvaluationSettings:
    """Configuration of the evaluation policy."""

    feasible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, bool):
            raise TypeError(
                "feasible must be a boolean."
            )

EVALUATION_SETTINGS = EvaluationSettings(
    feasible=True,
)


def _set_greedy_policy(
    agent: TabularQLearningAgent,
    evaluation_settings: EvaluationSettings,
) -> None:
    """
    Disable exploration and apply the evaluation action-space policy.

    The learned Q-table remains unchanged. Only the action-selection
    configuration is replaced.
    """
    learning_agent = agent.learner.agent

    learning_agent.settings = QLearningAgentSettings(
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        feasible=evaluation_settings.feasible,
    )


def _print_evaluation_results(
    *,
    agent: TabularQLearningAgent,
    metrics: QLearningExperimentMetrics,
    results,
    checkpoint_path: Path,
) -> None:
    """Print evaluation statistics."""

    completed_results = [
        result
        for result in results
        if result.outcome != EpisodeOutcome.INTERRUPTED
    ]

    success_count = sum(
        result.outcome == EpisodeOutcome.SUCCESS
        for result in completed_results
    )

    timeout_count = sum(
        result.outcome == EpisodeOutcome.HORIZON_TIMEOUT
        for result in completed_results
    )

    completed_count = len(completed_results)

    print()
    print("=" * 70)
    print("TABULAR Q-LEARNING EVALUATION")
    print("=" * 70)

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Evaluated episodes:",
        completed_count,
    )

    print(
        "Policy: greedy (epsilon=0)",
    )

    print()

    for result in results:
        print(
            f"Episode {result.episode_number}: "
            f"{result.outcome.value}, "
            f"decision points="
            f"{result.decision_points}"
        )

    print()
    print("Evaluation metrics:")

    print(
        "Simulation Time:",
        metrics.simulation_time
    )

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

    if completed_count > 0:
        print(
            "Success episodes:",
            success_count,
            f"({success_count / completed_count:.2%})",
        )

        print(
            "Horizon timeouts:",
            timeout_count,
            f"({timeout_count / completed_count:.2%})",
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


def evaluate(
    *,
    plant_settings: PlantSimulationSettings,
    tcp_settings: TCPSettings | None = None,
    num_episodes: int = 10,
    checkpoint_path: Path = CHECKPOINT_PATH,
    evaluation_settings: EvaluationSettings = EVALUATION_SETTINGS
) -> None:
    """
    Evaluate a learned tabular Q-learning policy.

    The checkpoint is restored, epsilon-greedy exploration is disabled,
    and no Q-learning updates are performed during evaluation.
    """

    if num_episodes <= 0:
        raise ValueError(
            "num_episodes must be positive."
        )

    if tcp_settings is None:
        tcp_settings = TCPSettings()

    checkpoint_settings = (
        QLearningCheckpointSettings(
            path=checkpoint_path,
        )
    )

    checkpoint_manager = (
        QLearningCheckpointManager(
            checkpoint_settings
        )
    )

    if not checkpoint_manager.exists:
        raise FileNotFoundError(
            "No Q-learning checkpoint exists at: "
            f"{checkpoint_path}"
        )

    problem = create_problem(
        EXPERIMENT_SETTINGS
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

    try:
        checkpoint = (
            checkpoint_manager.restore_from_checkpoint(
                agent=agent,
                experiment_settings=EXPERIMENT_SETTINGS,
            )
        )

        plotter = (
            QTablePlotter()
        )

        top_figure, _ = (
            plotter.plot_top_q_values(
                checkpoint.q_table,
                top_n=20,
            )
        )


        _set_greedy_policy(
            agent,
            evaluation_settings,
        )

        print(
            "Loaded checkpoint from:",
            checkpoint_path,
        )

        print(
            "Training episodes in checkpoint:",
            checkpoint.episode_count,
        )

        print(
            "Training decision points in checkpoint:",
            checkpoint.decision_count,
        )

        print(
            "Evaluation policy: epsilon = 0"
        )

        print()

        runner = QLearningEpisodeRunner(
            environment=environment,
            agent=agent,
            learn=False,
        )

        results, metrics = runner.run(
            num_episodes=num_episodes,
        )

        evaluation_state = agent.learner.last_q_state

        state_figures = None

        if evaluation_state is not None:
            state_figure, axes = (
                plotter.plot_state_action_values(
                    q_values=checkpoint.q_table,
                    state=evaluation_state,
                    actions=(0, 1, 2),
                )
            )

        plt.show()

        _print_evaluation_results(
            agent=agent,
            metrics=metrics,
            results=results,
            checkpoint_path=checkpoint_path,
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
        "Import evaluate() and provide PlantSimulationSettings "
        "from your application configuration."
    )
