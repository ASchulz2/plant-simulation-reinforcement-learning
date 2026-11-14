from __future__ import annotations

import pytest

from problem.observation import Observation
from problem.problem import ManufacturingProblem
from problem.tabular_state import TabularState


def make_observation(
    throughput: int,
) -> Observation:
    return Observation(
        simulation_time=0.0,
        admission_buffer_contents=(),
        storage_a_count=0,
        storage_a_language="",
        storage_b_count=0,
        storage_b_language="",
        throughput=throughput,
    )


def test_default_action_space() -> None:
    problem = ManufacturingProblem()

    assert problem.actions == (0, 1, 2)


def test_default_storage_capacity() -> None:
    problem = ManufacturingProblem()

    assert problem.storage_capacity == 10


def test_valid_actions_are_accepted() -> None:
    problem = ManufacturingProblem()

    for action in problem.actions:
        problem.validate_action(action)


def test_invalid_action_is_rejected() -> None:
    problem = ManufacturingProblem()

    with pytest.raises(ValueError):
        problem.validate_action(3)


def test_non_integer_action_is_rejected() -> None:
    problem = ManufacturingProblem()

    with pytest.raises(TypeError):
        problem.validate_action("2")  # type: ignore[arg-type]


def test_successful_terminal_condition() -> None:
    problem = ManufacturingProblem(
        target_throughput=1000,
    )

    assert not problem.is_successful(
        make_observation(999)
    )

    assert problem.is_successful(
        make_observation(1000)
    )

    assert problem.is_successful(
        make_observation(1001)
    )


def make_tabular_state(
    *,
    current_language: str = "GER",
    storage_a_language: str = "",
    storage_a_count: int = 0,
    storage_b_language: str = "",
    storage_b_count: int = 0,
) -> TabularState:
    return TabularState(
        current_language=current_language,
        admission_buffer_ger_count=1
        if current_language == "GER"
        else 0,
        admission_buffer_fr_count=1
        if current_language == "FR"
        else 0,
        admission_buffer_rus_count=1
        if current_language == "RUS"
        else 0,
        storage_a_language=storage_a_language,
        storage_a_count=storage_a_count,
        storage_b_language=storage_b_language,
        storage_b_count=storage_b_count,
    )


# ---------------------------------------------------------------------------
# Basic feasibility
# ---------------------------------------------------------------------------


def test_both_empty_storages_make_all_actions_feasible() -> None:
    problem = ManufacturingProblem()

    state = make_tabular_state()

    assert problem.feasible_actions(state) == (0, 1, 2)


def test_matching_storage_is_feasible() -> None:
    problem = ManufacturingProblem()

    state = make_tabular_state(
        storage_a_language="GER",
        storage_a_count=3,
        storage_b_language="FR",
        storage_b_count=2,
    )

    assert problem.feasible_actions(state) == (0, 2)


def test_non_matching_storage_is_not_feasible() -> None:
    problem = ManufacturingProblem()

    state = make_tabular_state(
        storage_a_language="FR",
        storage_a_count=3,
        storage_b_language="RUS",
        storage_b_count=2,
    )

    assert problem.feasible_actions(state) == (2,)


def test_empty_storage_is_feasible_for_any_language() -> None:
    problem = ManufacturingProblem()

    state = make_tabular_state(
        current_language="RUS",
        storage_a_language="",
        storage_a_count=0,
        storage_b_language="GER",
        storage_b_count=3,
    )

    assert problem.feasible_actions(state) == (0, 2)


def test_return_is_always_feasible_when_mu_exists() -> None:
    problem = ManufacturingProblem()

    for language in ("GER", "FR", "RUS"):
        state = make_tabular_state(
            current_language=language,
            storage_a_language="GER",
            storage_a_count=10,
            storage_b_language="FR",
            storage_b_count=10,
        )

        assert 2 in problem.feasible_actions(state)


def test_empty_current_language_has_no_feasible_actions() -> None:
    problem = ManufacturingProblem()

    state = make_tabular_state(
        current_language="",
        storage_a_language="GER",
        storage_a_count=5,
        storage_b_language="FR",
        storage_b_count=5,
    )

    assert problem.feasible_actions(state) == ()


# ---------------------------------------------------------------------------
# Capacity constraints
# ---------------------------------------------------------------------------


def test_matching_storage_below_capacity_is_feasible() -> None:
    problem = ManufacturingProblem(
        storage_capacity=10,
    )

    state = make_tabular_state(
        current_language="GER",
        storage_a_language="GER",
        storage_a_count=9,
        storage_b_language="FR",
        storage_b_count=3,
    )

    assert problem.feasible_actions(state) == (0, 2)


def test_matching_storage_at_capacity_is_not_feasible() -> None:
    problem = ManufacturingProblem(
        storage_capacity=10,
    )

    state = make_tabular_state(
        current_language="GER",
        storage_a_language="GER",
        storage_a_count=10,
        storage_b_language="FR",
        storage_b_count=3,
    )

    assert problem.feasible_actions(state) == (2,)


def test_empty_storage_at_zero_occupancy_is_feasible() -> None:
    problem = ManufacturingProblem(
        storage_capacity=10,
    )

    state = make_tabular_state(
        current_language="GER",
        storage_a_language="",
        storage_a_count=0,
        storage_b_language="GER",
        storage_b_count=10,
    )

    assert problem.feasible_actions(state) == (0, 2)


def test_non_matching_storage_at_capacity_is_not_feasible() -> None:
    problem = ManufacturingProblem(
        storage_capacity=10,
    )

    state = make_tabular_state(
        current_language="GER",
        storage_a_language="FR",
        storage_a_count=10,
        storage_b_language="GER",
        storage_b_count=10,
    )

    assert problem.feasible_actions(state) == (2,)


# ---------------------------------------------------------------------------
# Capacity configuration
# ---------------------------------------------------------------------------


def test_custom_storage_capacity_is_respected() -> None:
    problem = ManufacturingProblem(
        storage_capacity=5,
    )

    below_capacity = make_tabular_state(
        current_language="GER",
        storage_a_language="GER",
        storage_a_count=4,
    )

    at_capacity = make_tabular_state(
        current_language="GER",
        storage_a_language="GER",
        storage_a_count=5,
    )

    assert problem.feasible_actions(
        below_capacity
    ) == (0, 1, 2)

    assert problem.feasible_actions(
        at_capacity
    ) == (1, 2)


def test_storage_capacity_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="storage_capacity must be positive",
    ):
        ManufacturingProblem(
            storage_capacity=0,
        )


def test_negative_storage_capacity_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="storage_capacity must be positive",
    ):
        ManufacturingProblem(
            storage_capacity=-1,
        )
