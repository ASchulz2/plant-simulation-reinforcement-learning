from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class TrainingHistory:
    """Scalar metrics collected during PPO training."""

    iteration: list[int] = field(
        default_factory=list
    )

    # --------------------------------------------------------------
    # Environment performance
    # --------------------------------------------------------------

    reward: list[float] = field(
        default_factory=list
    )

    rollout_simulation_time: list[float] = field(
        default_factory=list
    )

    cumulative_simulation_time: list[float] = field(
        default_factory=list
    )

    throughput: list[int] = field(
        default_factory=list
    )

    throughput_increment: list[int] = field(
        default_factory=list
    )

    # --------------------------------------------------------------
    # PPO optimization
    # --------------------------------------------------------------

    policy_loss: list[float] = field(
        default_factory=list
    )

    value_loss: list[float] = field(
        default_factory=list
    )

    total_loss: list[float] = field(
        default_factory=list
    )

    entropy: list[float] = field(
        default_factory=list
    )

    approximate_kl: list[float] = field(
        default_factory=list
    )

    clip_fraction: list[float] = field(
        default_factory=list
    )

    # --------------------------------------------------------------
    # GAE diagnostics
    # --------------------------------------------------------------

    advantage_mean: list[float] = field(
        default_factory=list
    )

    advantage_std: list[float] = field(
        default_factory=list
    )

    return_mean: list[float] = field(
        default_factory=list
    )

    return_std: list[float] = field(
        default_factory=list
    )

    def append(
        self,
        *,
        iteration: int,
        reward: float,
        rollout_simulation_time: float,
        cumulative_simulation_time: float,
        throughput: int,
        throughput_increment: int,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        entropy: float,
        approximate_kl: float,
        clip_fraction: float,
        advantage_mean: float,
        advantage_std: float,
        return_mean: float,
        return_std: float,
    ) -> None:
        """Append one PPO training observation."""

        self.iteration.append(iteration)

        # Environment performance
        self.reward.append(reward)
        self.rollout_simulation_time.append(
            rollout_simulation_time
        )
        self.cumulative_simulation_time.append(
            cumulative_simulation_time
        )
        self.throughput.append(throughput)
        self.throughput_increment.append(
            throughput_increment
        )

        # PPO diagnostics
        self.policy_loss.append(policy_loss)
        self.value_loss.append(value_loss)
        self.total_loss.append(total_loss)
        self.entropy.append(entropy)
        self.approximate_kl.append(
            approximate_kl
        )
        self.clip_fraction.append(
            clip_fraction
        )

        # GAE diagnostics
        self.advantage_mean.append(
            advantage_mean
        )
        self.advantage_std.append(
            advantage_std
        )
        self.return_mean.append(return_mean)
        self.return_std.append(return_std)


@dataclass
class VisualizationTool:
    """Visualize PPO training and manufacturing performance."""

    history: TrainingHistory = field(
        default_factory=TrainingHistory
    )

    def record(
        self,
        *,
        iteration: int,
        reward: float,
        rollout_simulation_time: float,
        cumulative_simulation_time: float,
        throughput: int,
        throughput_increment: int,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        entropy: float,
        approximate_kl: float,
        clip_fraction: float,
        advantage_mean: float,
        advantage_std: float,
        return_mean: float,
        return_std: float,
    ) -> None:
        """Record metrics from one PPO update."""

        self.history.append(
            iteration=iteration,
            reward=reward,
            rollout_simulation_time=(
                rollout_simulation_time
            ),
            cumulative_simulation_time=(
                cumulative_simulation_time
            ),
            throughput=throughput,
            throughput_increment=(
                throughput_increment
            ),
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            entropy=entropy,
            approximate_kl=approximate_kl,
            clip_fraction=clip_fraction,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            return_mean=return_mean,
            return_std=return_std,
        )

    # ==================================================================
    # Environment performance
    # ==================================================================

    def plot_reward(self) -> None:
        """Plot reward per rollout."""

        self._plot(
            self.history.iteration,
            self.history.reward,
            title="Reward per Rollout",
            xlabel="Training iteration",
            ylabel="Reward",
        )

    def plot_simulation_time(self) -> None:
        """
        Plot simulation time consumed by each rollout.

        This is the primary training-performance metric for the
        throughput-time objective.
        """

        self._plot(
            self.history.iteration,
            self.history.rollout_simulation_time,
            title="Simulation Time per Rollout",
            xlabel="Training iteration",
            ylabel="Simulation time",
        )

    def plot_cumulative_simulation_time(self) -> None:
        """Plot cumulative simulation time."""

        self._plot(
            self.history.iteration,
            self.history.cumulative_simulation_time,
            title="Cumulative Simulation Time",
            xlabel="Training iteration",
            ylabel="Cumulative simulation time",
        )

    def plot_throughput(self) -> None:
        """
        Plot cumulative production throughput.

        Throughput is measured as the number of completed
        games at the end of each rollout.
        """

        self._plot(
            self.history.iteration,
            self.history.throughput,
            title="Cumulative Throughput",
            xlabel="Training iteration",
            ylabel="Completed games",
        )

    def plot_throughput_increment(self) -> None:
        """
        Plot production progress per rollout.

        This shows how many additional games were completed
        during each rollout.
        """

        self._plot(
            self.history.iteration,
            self.history.throughput_increment,
            title="Throughput per Rollout",
            xlabel="Training iteration",
            ylabel="Games completed",
        )

    # ==================================================================
    # PPO diagnostics
    # ==================================================================

    def plot_losses(self) -> None:
        """Plot policy, value, and total losses."""

        fig, ax = plt.subplots()

        ax.plot(
            self.history.iteration,
            self.history.policy_loss,
            label="Policy loss",
        )

        ax.plot(
            self.history.iteration,
            self.history.value_loss,
            label="Value loss",
        )

        ax.plot(
            self.history.iteration,
            self.history.total_loss,
            label="Total loss",
        )

        ax.set_xlabel(
            "Training iteration"
        )
        ax.set_ylabel("Loss")
        ax.set_title("PPO Losses")
        ax.legend()

        fig.tight_layout()

    def plot_policy_diagnostics(self) -> None:
        """Plot KL divergence and clipping fraction."""

        fig, ax = plt.subplots()

        ax.plot(
            self.history.iteration,
            self.history.approximate_kl,
            label="Approximate KL",
        )

        ax.plot(
            self.history.iteration,
            self.history.clip_fraction,
            label="Clip fraction",
        )

        ax.set_xlabel(
            "Training iteration"
        )
        ax.set_ylabel("Value")
        ax.set_title(
            "PPO Policy Diagnostics"
        )
        ax.legend()

        fig.tight_layout()

    def plot_entropy(self) -> None:
        """Plot policy entropy."""

        self._plot(
            self.history.iteration,
            self.history.entropy,
            title="Policy Entropy",
            xlabel="Training iteration",
            ylabel="Entropy",
        )

    def plot_advantages(self) -> None:
        """Plot GAE advantage statistics."""

        fig, ax = plt.subplots()

        ax.plot(
            self.history.iteration,
            self.history.advantage_mean,
            label="Mean",
        )

        ax.plot(
            self.history.iteration,
            self.history.advantage_std,
            label="Std",
        )

        ax.set_xlabel(
            "Training iteration"
        )
        ax.set_ylabel("Advantage")
        ax.set_title("GAE Diagnostics")
        ax.legend()

        fig.tight_layout()

    # ==================================================================
    # Combined visualization
    # ==================================================================

    def show_training(self) -> None:
        """Display the main training diagnostics."""

        self.plot_reward()
        self.plot_simulation_time()
        self.plot_throughput()
        self.plot_throughput_increment()
        self.plot_losses()
        self.plot_policy_diagnostics()
        self.plot_entropy()
        self.plot_advantages()

        plt.show()

    # ==================================================================
    # Internal plotting helper
    # ==================================================================

    @staticmethod
    def _plot(
        x: list[int],
        y: list[float] | list[int],
        *,
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> None:
        """Create a single training plot."""

        fig, ax = plt.subplots()

        ax.plot(
            x,
            y,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        fig.tight_layout()
