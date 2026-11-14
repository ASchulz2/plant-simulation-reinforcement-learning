from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from problem.tabular_state import TabularState
from dataclasses import astuple

_ACTION_LABELS = {
    0: "A",
    1: "B",
    2: "R",
}

class QTablePlotter:
    """
    Visualize selected parts of a learned tabular Q-function.

    The plotter consumes a snapshot of explicitly stored Q-values and
    has no knowledge of Plant Simulation, the environment, or training.
    """

    def plot_top_q_values(
        self,
        q_values: Mapping[
            tuple[TabularState, int],
            float,
        ],
        *,
        top_n: int = 20,
    ) -> tuple[Figure, Axes]:
        """
        Plot the state-action pairs with the largest absolute Q-values.
        """
        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        ranked = sorted(
            q_values.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]

        labels = [
            (
                f"{astuple(state)} | "
                f"{_ACTION_LABELS.get(action, str(action))}"
            )
            for (state, action), _ in ranked
        ]

        values = [
            value
            for _, value in ranked
        ]

        figure, axes = plt.subplots()

        axes.barh(
            range(len(values)),
            values,
        )

        axes.set_yticks(range(len(labels)))
        axes.set_yticklabels(
            labels,
            fontsize=8,
        )
        axes.invert_yaxis()

        axes.set_xlabel("Q-value")
        axes.set_ylabel("State | Action")
        axes.set_title(
            f"Top {len(values)} Learned State-Action Values"
        )
        axes.grid(True, axis="x")

        figure.tight_layout()

        return figure, axes

    def plot_state_action_values(
        self,
        q_values: Mapping[
            tuple[TabularState, int],
            float,
        ],
        state: TabularState,
        actions: tuple[int, ...] = (0, 1, 2),
    ) -> tuple[Figure, Axes]:
        """
        Plot Q-values for the specified actions in one state.
        """

        values = [
            q_values.get(
                (state, action),
                0.0,
            )
            for action in actions
        ]

        labels = [
            _ACTION_LABELS.get(
                action,
                str(action),
            )
            for action in actions
        ]

        figure, axes = plt.subplots()

        axes.bar(
            labels,
            values,
        )

        axes.set_xlabel("Action")
        axes.set_ylabel("Q-value")
        axes.set_title(
            f"Learned Q-Values | State = {astuple(state)}"
        )
        axes.grid(True, axis="y")

        figure.tight_layout()

        return figure, axes