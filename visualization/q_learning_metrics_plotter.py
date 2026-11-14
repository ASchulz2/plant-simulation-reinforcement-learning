from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from environment.q_learning_metrics import (
    QLearningExperimentMetrics,
)


class QLearningMetricsPlotter:
    """
    Visualize per-episode metrics from a tabular Q-learning experiment.

    This class only consumes experiment metrics. It has no knowledge of
    Plant Simulation, the environment, or the Q-learning agent.
    """

    def plot_total_reward(
        self,
        metrics: QLearningExperimentMetrics,
    ) -> tuple[Figure, Axes]:
        episodes = [
            episode.episode_number
            for episode in metrics.episodes
        ]
        rewards = [
            episode.total_reward
            for episode in metrics.episodes
        ]

        figure, axes = plt.subplots()

        axes.plot(
            episodes,
            rewards,
            marker="o",
        )
        axes.set_xlabel("Episode")
        axes.set_ylabel("Total reward")
        axes.set_title(
            "Q-learning: Total Reward per Episode"
        )
        axes.grid(True)

        figure.tight_layout()

        return figure, axes

    def plot_throughput(
        self,
        metrics: QLearningExperimentMetrics,
    ) -> tuple[Figure, Axes]:
        episodes = [
            episode.episode_number
            for episode in metrics.episodes
        ]
        throughput = [
            episode.throughput
            for episode in metrics.episodes
        ]

        figure, axes = plt.subplots()

        axes.plot(
            episodes,
            throughput,
            marker="o",
        )
        axes.set_xlabel("Episode")
        axes.set_ylabel("Throughput")
        axes.set_title(
            "Q-learning: Throughput per Episode"
        )
        axes.grid(True)

        figure.tight_layout()

        return figure, axes

    def plot_decision_points(
        self,
        metrics: QLearningExperimentMetrics,
    ) -> tuple[Figure, Axes]:
        episodes = [
            episode.episode_number
            for episode in metrics.episodes
        ]
        decision_points = [
            episode.decision_points
            for episode in metrics.episodes
        ]

        figure, axes = plt.subplots()

        axes.plot(
            episodes,
            decision_points,
            marker="o",
        )
        axes.set_xlabel("Episode")
        axes.set_ylabel("Decision points")
        axes.set_title(
            "Q-learning: Decision Points per Episode"
        )
        axes.grid(True)

        figure.tight_layout()

        return figure, axes

    def plot_mean_drain_count(
        self,
        metrics: QLearningExperimentMetrics,
    ) -> tuple[Figure, Axes]:
        episodes = [
            episode.episode_number
            for episode in metrics.episodes
        ]
        mean_drain_counts = [
            episode.mean_drain_count
            for episode in metrics.episodes
        ]

        figure, axes = plt.subplots()

        axes.plot(
            episodes,
            mean_drain_counts,
            marker="o",
        )
        axes.set_xlabel("Episode")
        axes.set_ylabel("Mean MUs per emptying event")
        axes.set_title(
            "Q-learning: Mean Emptying Count per Episode"
        )
        axes.grid(True)

        figure.tight_layout()

        return figure, axes

    def plot_simulation_time(
        self,
        metrics: QLearningExperimentMetrics,
    ) -> tuple[Figure, Axes]:
        """
        Plot the Plant Simulation time for each Q-learning episode.

        Parameters
        ----------
        metrics:
            Experiment metrics containing one entry per completed episode.

        Returns
        -------
        tuple[Figure, Axes]
            The Matplotlib figure and axes containing the simulation-time plot.
        """
        episodes = [
            episode.episode_number
            for episode in metrics.episodes
        ]
        simulation_times = [
            episode.simulation_time
            for episode in metrics.episodes
        ]

        figure, axes = plt.subplots()

        axes.plot(
            episodes,
            simulation_times,
            marker="o",
        )
        axes.set_xlabel("Episode")
        axes.set_ylabel("Simulation time")
        axes.set_title(
            "Q-learning: Simulation Time per Episode"
        )
        axes.grid(True)

        figure.tight_layout()

        return figure, axes