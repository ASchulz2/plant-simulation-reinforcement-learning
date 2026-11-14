from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from learning.ppo_trainer import PPOUpdateResult


@dataclass
class PPOVisualizer:
    """
    Visualize PPO training diagnostics across rollout updates.

    One entry is recorded for every call to PPOTrainer.update().
    """

    update_results: list[PPOUpdateResult] = field(
        default_factory=list
    )

    def record(
        self,
        result: PPOUpdateResult,
    ) -> None:
        """Record the result of one PPO update."""

        self.update_results.append(
            result
        )

    @property
    def num_updates(self) -> int:
        """Number of recorded PPO updates."""

        return len(
            self.update_results
        )

    def clear(self) -> None:
        """Remove all recorded PPO updates."""

        self.update_results.clear()

    def plot_losses(
        self,
    ) -> None:
        """Plot policy, value, and total loss."""

        self._require_data()

        x = self._x()

        policy_loss = [
            result.policy_loss
            for result in self.update_results
        ]

        value_loss = [
            result.value_loss
            for result in self.update_results
        ]

        total_loss = [
            result.total_loss
            for result in self.update_results
        ]

        plt.figure()

        plt.plot(
            x,
            policy_loss,
            label="Policy loss",
        )

        plt.plot(
            x,
            value_loss,
            label="Value loss",
        )

        plt.plot(
            x,
            total_loss,
            label="Total loss",
        )

        plt.xlabel("PPO update")
        plt.ylabel("Loss")
        plt.title("PPO Losses")
        plt.legend()
        plt.grid(True)

        plt.show()

    def plot_entropy(
        self,
    ) -> None:
        """Plot policy entropy."""

        self._require_data()

        x = self._x()

        entropy = [
            result.entropy
            for result in self.update_results
        ]

        plt.figure()

        plt.plot(
            x,
            entropy,
        )

        plt.xlabel("PPO update")
        plt.ylabel("Entropy")
        plt.title("Policy Entropy")
        plt.grid(True)

        plt.show()

    def plot_kl(
        self,
    ) -> None:
        """Plot approximate KL divergence."""

        self._require_data()

        x = self._x()

        approximate_kl = [
            result.approximate_kl
            for result in self.update_results
        ]

        plt.figure()

        plt.plot(
            x,
            approximate_kl,
        )

        plt.axhline(
            0.0,
            linestyle="--",
        )

        plt.xlabel("PPO update")
        plt.ylabel("Approximate KL")
        plt.title("PPO Approximate KL")
        plt.grid(True)

        plt.show()

    def plot_clip_fraction(
        self,
    ) -> None:
        """Plot PPO clipping fraction."""

        self._require_data()

        x = self._x()

        clip_fraction = [
            result.clip_fraction
            for result in self.update_results
        ]

        plt.figure()

        plt.plot(
            x,
            clip_fraction,
        )

        plt.xlabel("PPO update")
        plt.ylabel("Clip fraction")
        plt.title("PPO Clip Fraction")
        plt.grid(True)

        plt.show()

    def plot_advantages(
        self,
    ) -> None:
        """Plot mean and standard deviation of advantages."""

        self._require_data()

        x = self._x()

        means = [
            result.advantage_mean
            for result in self.update_results
        ]

        stds = [
            result.advantage_std
            for result in self.update_results
        ]

        plt.figure()

        plt.plot(
            x,
            means,
            label="Mean",
        )

        plt.plot(
            x,
            stds,
            label="Std",
        )

        plt.xlabel("PPO update")
        plt.ylabel("Advantage")
        plt.title("GAE Advantage Statistics")
        plt.legend()
        plt.grid(True)

        plt.show()

    def plot_returns(
        self,
    ) -> None:
        """Plot mean and standard deviation of returns."""

        self._require_data()

        x = self._x()

        means = [
            result.return_mean
            for result in self.update_results
        ]

        stds = [
            result.return_std
            for result in self.update_results
        ]

        plt.figure()

        plt.plot(
            x,
            means,
            label="Mean",
        )

        plt.plot(
            x,
            stds,
            label="Std",
        )

        plt.xlabel("PPO update")
        plt.ylabel("Return")
        plt.title("PPO Return Statistics")
        plt.legend()
        plt.grid(True)

        plt.show()

    def plot_all(
        self,
    ) -> None:
        """
        Display all PPO diagnostic plots.

        Each diagnostic receives its own figure.
        """

        self.plot_losses()
        self.plot_entropy()
        self.plot_kl()
        self.plot_clip_fraction()
        self.plot_advantages()
        self.plot_returns()

    def _x(self) -> list[int]:
        """Return 1-based PPO update indices."""

        return list(
            range(
                1,
                self.num_updates + 1,
            )
        )

    def _require_data(self) -> None:
        """Raise if no PPO results have been recorded."""

        if not self.update_results:
            raise RuntimeError(
                "No PPO update results available. "
                "Call record() first."
            )