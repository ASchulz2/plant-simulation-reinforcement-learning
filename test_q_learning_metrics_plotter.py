from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from environment.episode import EpisodeOutcome
from environment.q_learning_metrics import (
    QLearningEpisodeMetrics,
    QLearningExperimentMetrics,
)
from visualization.q_learning_metrics_plotter import (
    QLearningMetricsPlotter,
)


def create_metrics() -> QLearningExperimentMetrics:
    metrics = QLearningExperimentMetrics()

    metrics.add(
        QLearningEpisodeMetrics(
            episode_number=1,
            outcome=EpisodeOutcome.HORIZON_TIMEOUT,
            decision_points=100,
            simulation_time=20.0,
            throughput=5,
            total_reward=10.0,
            action_counts=(30, 40, 30),
            drain_event_count=2,
            drained_mu_count=10,
        )
    )

    metrics.add(
        QLearningEpisodeMetrics(
            episode_number=2,
            outcome=EpisodeOutcome.HORIZON_TIMEOUT,
            decision_points=90,
            simulation_time=18.0,
            throughput=8,
            total_reward=15.0,
            action_counts=(30, 40, 30),
            drain_event_count=2,
            drained_mu_count=14,
        )
    )

    metrics.add(
        QLearningEpisodeMetrics(
            episode_number=3,
            outcome=EpisodeOutcome.SUCCESS,
            decision_points=70,
            simulation_time=14.0,
            throughput=12,
            total_reward=25.0,
            action_counts=(30, 40, 30),
            drain_event_count=2,
            drained_mu_count=18,
        )
    )

    return metrics


def test_plot_total_reward_uses_episode_history():
    plotter = QLearningMetricsPlotter()
    metrics = create_metrics()

    figure, axes = plotter.plot_total_reward(
        metrics
    )

    line = axes.lines[0]

    assert list(line.get_xdata()) == [1, 2, 3]
    assert list(line.get_ydata()) == [
        10.0,
        15.0,
        25.0,
    ]

    assert axes.get_xlabel() == "Episode"
    assert axes.get_ylabel() == "Total reward"

    figure.clf()


def test_plot_throughput_uses_episode_history():
    plotter = QLearningMetricsPlotter()
    metrics = create_metrics()

    figure, axes = plotter.plot_throughput(
        metrics
    )

    line = axes.lines[0]

    assert list(line.get_xdata()) == [1, 2, 3]
    assert list(line.get_ydata()) == [
        5,
        8,
        12,
    ]

    assert axes.get_xlabel() == "Episode"
    assert axes.get_ylabel() == "Throughput"

    figure.clf()


def test_plot_decision_points_uses_episode_history():
    plotter = QLearningMetricsPlotter()
    metrics = create_metrics()

    figure, axes = plotter.plot_decision_points(
        metrics
    )

    line = axes.lines[0]

    assert list(line.get_xdata()) == [1, 2, 3]
    assert list(line.get_ydata()) == [
        100,
        90,
        70,
    ]

    assert axes.get_xlabel() == "Episode"
    assert axes.get_ylabel() == "Decision points"

    figure.clf()


def test_plot_empty_metrics_creates_empty_plot():
    plotter = QLearningMetricsPlotter()
    metrics = QLearningExperimentMetrics()

    figure, axes = plotter.plot_total_reward(
        metrics
    )

    assert len(axes.lines) == 1
    assert list(axes.lines[0].get_xdata()) == []
    assert list(axes.lines[0].get_ydata()) == []

    figure.clf()

def test_plot_mean_drain_count_uses_episode_history():
    plotter = QLearningMetricsPlotter()
    metrics = create_metrics()

    figure, axes = plotter.plot_mean_drain_count(
        metrics
    )

    line = axes.lines[0]

    assert list(line.get_xdata()) == [1, 2, 3]
    assert list(line.get_ydata()) == [
        5.0,
        7.0,
        9.0,
    ]

    assert axes.get_xlabel() == "Episode"
    assert axes.get_ylabel() == "Mean MUs per emptying event"

    figure.clf()

def test_plot_empty_metrics_creates_empty_mean_drain_count_plot():
    plotter = QLearningMetricsPlotter()
    metrics = QLearningExperimentMetrics()

    figure, axes = plotter.plot_mean_drain_count(
        metrics
    )

    assert len(axes.lines) == 1
    assert list(axes.lines[0].get_xdata()) == []
    assert list(axes.lines[0].get_ydata()) == []

    figure.clf()