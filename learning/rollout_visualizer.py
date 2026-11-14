from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt

from environment.rollout import Rollout


@dataclass
class RolloutVisualizer:
    """
    Visualize the temporal structure of one rollout.

    The visualizer operates directly on Transition data and is
    independent of PPO optimization.
    """

    def plot_actions(
        self,
        rollout: Rollout,
    ) -> None:
        """Plot the selected action at every decision."""

        self._require_non_empty(
            rollout
        )

        actions = [
            transition.action
            for transition in rollout.transitions
        ]

        steps = list(
            range(
                1,
                len(actions) + 1,
            )
        )

        fig, ax = plt.subplots()

        ax.step(
            steps,
            actions,
            where="post",
        )

        ax.set_xlabel("Decision")
        ax.set_ylabel("Action")
        ax.set_title("Actions per Decision")
        ax.set_yticks([0, 1, 2])

        fig.tight_layout()

    def plot_rewards(
        self,
        rollout: Rollout,
    ) -> None:
        """Plot reward per transition."""

        self._require_non_empty(
            rollout
        )

        rewards = [
            transition.reward
            for transition in rollout.transitions
        ]

        steps = list(
            range(
                1,
                len(rewards) + 1,
            )
        )

        fig, ax = plt.subplots()

        ax.plot(
            steps,
            rewards,
        )

        ax.axhline(
            0.0,
            linestyle="--",
        )

        ax.set_xlabel("Decision")
        ax.set_ylabel("Reward")
        ax.set_title("Reward per Decision")

        fig.tight_layout()

    def plot_cumulative_reward(
        self,
        rollout: Rollout,
    ) -> None:
        """Plot cumulative reward across the rollout."""

        self._require_non_empty(
            rollout
        )

        cumulative_rewards: list[float] = []

        total = 0.0

        for transition in rollout.transitions:

            total += transition.reward
            cumulative_rewards.append(
                total
            )

        steps = list(
            range(
                1,
                len(cumulative_rewards) + 1,
            )
        )

        fig, ax = plt.subplots()

        ax.plot(
            steps,
            cumulative_rewards,
        )

        ax.set_xlabel("Decision")
        ax.set_ylabel("Cumulative reward")
        ax.set_title("Cumulative Reward")

        fig.tight_layout()

    def plot_simulation_time(
        self,
        rollout: Rollout,
    ) -> None:
        """Plot cumulative Plant Simulation time."""

        self._require_non_empty(
            rollout
        )

        cumulative_time: list[float] = []

        total = 0.0

        for transition in rollout.transitions:

            total += transition.elapsed_time
            cumulative_time.append(
                total
            )

        steps = list(
            range(
                1,
                len(cumulative_time) + 1,
            )
        )

        fig, ax = plt.subplots()

        ax.plot(
            steps,
            cumulative_time,
        )

        ax.set_xlabel("Decision")
        ax.set_ylabel("Simulation time")
        ax.set_title("Cumulative Simulation Time")

        fig.tight_layout()

    def plot_throughput(
        self,
        rollout: Rollout,
    ) -> None:
        """
        Plot manufacturing throughput over the rollout.

        Each transition contains the throughput before and after
        the corresponding decision.
        """

        self._require_non_empty(
            rollout
        )

        states = [
            transition.throughput
            for transition in rollout.transitions
        ]

        final_throughputs = [
            transition.next_throughput
            for transition in rollout.transitions
        ]

        steps = list(
            range(
                1,
                len(states) + 1,
            )
        )

        fig, ax = plt.subplots()

        ax.step(
            steps,
            states,
            where="pre",
            label="Throughput",
        )

        ax.step(
            steps,
            final_throughputs,
            where="post",
            linestyle="--",
            label="Next throughput",
        )

        ax.set_xlabel("Decision")
        ax.set_ylabel("Completed games")
        ax.set_title("Manufacturing Throughput")
        ax.legend()

        fig.tight_layout()

    def plot_throughput_increment(
        self,
        rollout: Rollout,
    ) -> None:
        """Plot throughput gained by each transition."""

        self._require_non_empty(
            rollout
        )

        increments = [
            (
                transition.next_throughput
                - transition.throughput
            )
            for transition in rollout.transitions
        ]

        steps = list(
            range(
                1,
                len(increments) + 1,
            )
        )

        fig, ax = plt.subplots()

        ax.bar(
            steps,
            increments,
        )

        ax.set_xlabel("Decision")
        ax.set_ylabel("Completed games")
        ax.set_title("Throughput Increment per Decision")

        fig.tight_layout()

    def show_rollout(
        self,
        rollout: Rollout,
    ) -> None:
        """Display the main rollout diagnostics."""

        self._require_non_empty(
            rollout
        )

        self.plot_actions(
            rollout
        )

        self.plot_rewards(
            rollout
        )

        self.plot_cumulative_reward(
            rollout
        )

        self.plot_simulation_time(
            rollout
        )

        self.plot_throughput(
            rollout
        )

        self.plot_throughput_increment(
            rollout
        )

        plt.show()

    def summarize(
        self,
        rollout: Rollout,
    ) -> None:
        """Print a compact rollout summary."""

        self._require_non_empty(
            rollout
        )

        transitions = rollout.transitions

        rewards = [
            transition.reward
            for transition in transitions
        ]

        elapsed_times = [
            transition.elapsed_time
            for transition in transitions
        ]

        actions = [
            transition.action
            for transition in transitions
        ]

        throughput = transitions[0].throughput
        final_throughput = (
            transitions[-1].next_throughput
        )

        total_simulation_time = sum(
            elapsed_times
        )

        terminated = any(
            transition.terminated
            for transition in transitions
        )

        print()
        print("=" * 70)
        print("ROLLOUT SUMMARY")
        print("=" * 70)

        print(
            "Transitions:",
            len(transitions),
        )

        print(
            "Initial throughput:",
            throughput,
        )

        print(
            "Final throughput:",
            final_throughput,
        )

        print(
            "Throughput increment:",
            final_throughput - throughput,
        )

        print(
            "Total reward:",
            sum(rewards),
        )

        print(
            "Mean reward:",
            sum(rewards) / len(rewards),
        )

        print(
            "Simulation time:",
            total_simulation_time,
        )

        print(
            "Action 0 count:",
            actions.count(0),
        )

        print(
            "Action 1 count:",
            actions.count(1),
        )

        print(
            "Action 2 count:",
            actions.count(2),
        )

        print(
            "Terminated:",
            terminated,
        )

    @staticmethod
    def _require_non_empty(
        rollout: Rollout,
    ) -> None:
        """Require a non-empty rollout."""

        if rollout.is_empty:
            raise ValueError(
                "Cannot visualize an empty rollout."
            )
