from __future__ import annotations

from dataclasses import dataclass
import matplotlib.pyplot as plt
import pytest

from visualization.q_table_plotter import QTablePlotter


@dataclass(frozen=True)
class _PlotState:
    current_language: str
    admission_counts: tuple[int, int, int]
    storage_a_language: str
    storage_a_count: int
    storage_b_language: str
    storage_b_count: int


def _state(
    *,
    current_language: str = "GER",
    admission_counts: tuple[int, int, int] = (3, 2, 1),
    storage_a_language: str = "GER",
    storage_a_count: int = 8,
    storage_b_language: str = "FR",
    storage_b_count: int = 4,
) -> _PlotState:
    return _PlotState(
        current_language=current_language,
        admission_counts=admission_counts,
        storage_a_language=storage_a_language,
        storage_a_count=storage_a_count,
        storage_b_language=storage_b_language,
        storage_b_count=storage_b_count,
    )


def test_plot_top_q_values_uses_largest_absolute_values() -> None:
    state_a = _state()
    state_b = _state(
        current_language="FR",
        admission_counts=(1, 4, 0),
    )

    q_values = {
        (state_a, 0): 1.0,
        (state_a, 1): -10.0,
        (state_a, 2): 3.0,
        (state_b, 0): 7.0,
    }

    plotter = QTablePlotter()

    figure, axes = plotter.plot_top_q_values(
        q_values,
        top_n=3,
    )

    try:
        labels = [
            label.get_text()
            for label in axes.get_yticklabels()
        ]

        values = [
            patch.get_width()
            for patch in axes.patches
        ]

        assert len(axes.patches) == 3

        assert "| B" in labels[0]
        assert values[0] == -10.0

        assert "| A" in labels[1]
        assert values[1] == 7.0

        assert "| R" in labels[2]
        assert values[2] == 3.0

    finally:
        plt.close(figure)


def test_plot_top_q_values_uses_compact_action_labels() -> None:
    state = _state()

    q_values = {
        (state, 0): 1.0,
        (state, 1): 2.0,
        (state, 2): 3.0,
    }

    plotter = QTablePlotter()

    figure, axes = plotter.plot_top_q_values(
        q_values,
        top_n=3,
    )

    try:
        labels = [
            label.get_text()
            for label in axes.get_yticklabels()
        ]

        assert any("| A" in label for label in labels)
        assert any("| B" in label for label in labels)
        assert any("| R" in label for label in labels)

        assert all(
            "Storage A" not in label
            and "Storage B" not in label
            and "Return" not in label
            for label in labels
        )

    finally:
        plt.close(figure)


def test_plot_top_q_values_rejects_non_positive_top_n() -> None:
    plotter = QTablePlotter()

    with pytest.raises(
        ValueError,
        match="top_n must be positive",
    ):
        plotter.plot_top_q_values(
            {},
            top_n=0,
        )


def test_plot_state_action_values_returns_stored_values() -> None:
    state = _state()

    q_values = {
        (state, 0): 4.5,
        (state, 1): 2.0,
        (state, 2): -1.5,
    }

    plotter = QTablePlotter()

    figure, axes = plotter.plot_state_action_values(
        q_values=q_values,
        state=state,
        actions=(0, 1, 2),
    )

    try:
        labels = [
            tick.get_text()
            for tick in axes.get_xticklabels()
        ]

        values = [
            patch.get_height()
            for patch in axes.patches
        ]

        assert labels == ["A", "B", "R"]
        assert values == [4.5, 2.0, -1.5]

    finally:
        plt.close(figure)


def test_plot_state_action_values_uses_zero_for_missing_values() -> None:
    state = _state()

    q_values = {
        (state, 0): 4.5,
    }

    plotter = QTablePlotter()

    figure, axes = plotter.plot_state_action_values(
        q_values=q_values,
        state=state,
        actions=(0, 1, 2),
    )

    try:
        values = [
            patch.get_height()
            for patch in axes.patches
        ]

        assert values == [4.5, 0.0, 0.0]

    finally:
        plt.close(figure)


def test_plot_state_action_values_uses_compact_title() -> None:
    state = _state()

    plotter = QTablePlotter()

    figure, axes = plotter.plot_state_action_values(
        q_values={
            (state, 0): 1.0,
            (state, 1): 2.0,
            (state, 2): 3.0,
        },
        state=state,
    )

    try:
        title = axes.get_title()

        assert "Learned Q-Values" in title
        assert "State =" in title
        assert "GER" in title

    finally:
        plt.close(figure)