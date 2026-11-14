from __future__ import annotations

import pytest

from problem.tabular_state import TabularState
from problem.tabular_state_space import TabularStateSpace


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_max_buffer_capacity_must_be_integer() -> None:
    with pytest.raises(
        TypeError,
        match="max_buffer_capacity must be an integer",
    ):
        TabularStateSpace(
            max_buffer_capacity=1.5,
            storage_capacity=2,
        )


def test_max_buffer_capacity_must_be_non_negative() -> None:
    with pytest.raises(
        ValueError,
        match="max_buffer_capacity must be non-negative",
    ):
        TabularStateSpace(
            max_buffer_capacity=-1,
            storage_capacity=2,
        )


def test_storage_capacity_must_be_integer() -> None:
    with pytest.raises(
        TypeError,
        match="storage_capacity must be an integer",
    ):
        TabularStateSpace(
            max_buffer_capacity=2,
            storage_capacity=1.5,
        )


def test_storage_capacity_must_be_non_negative() -> None:
    with pytest.raises(
        ValueError,
        match="storage_capacity must be non-negative",
    ):
        TabularStateSpace(
            max_buffer_capacity=2,
            storage_capacity=-1,
        )


# ---------------------------------------------------------------------------
# Buffer state enumeration
# ---------------------------------------------------------------------------


def test_empty_buffer_has_exactly_one_configuration() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=0,
        storage_capacity=0,
    )

    states = state_space.states()

    assert len(states) == 1

    state = states[0]

    assert state.current_language == ""

    assert state.admission_buffer_ger_count == 0
    assert state.admission_buffer_fr_count == 0
    assert state.admission_buffer_rus_count == 0

    assert state.storage_a_language == ""
    assert state.storage_a_count == 0

    assert state.storage_b_language == ""
    assert state.storage_b_count == 0


def test_single_buffer_mu_creates_three_current_language_states() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=1,
        storage_capacity=0,
    )

    states = state_space.states()

    assert len(states) == 4

    assert TabularState(
        current_language="",
        admission_buffer_ger_count=0,
        admission_buffer_fr_count=0,
        admission_buffer_rus_count=0,
        storage_a_language="",
        storage_a_count=0,
        storage_b_language="",
        storage_b_count=0,
    ) in states

    assert TabularState(
        current_language="GER",
        admission_buffer_ger_count=1,
        admission_buffer_fr_count=0,
        admission_buffer_rus_count=0,
        storage_a_language="",
        storage_a_count=0,
        storage_b_language="",
        storage_b_count=0,
    ) in states

    assert TabularState(
        current_language="FR",
        admission_buffer_ger_count=0,
        admission_buffer_fr_count=1,
        admission_buffer_rus_count=0,
        storage_a_language="",
        storage_a_count=0,
        storage_b_language="",
        storage_b_count=0,
    ) in states

    assert TabularState(
        current_language="RUS",
        admission_buffer_ger_count=0,
        admission_buffer_fr_count=0,
        admission_buffer_rus_count=1,
        storage_a_language="",
        storage_a_count=0,
        storage_b_language="",
        storage_b_count=0,
    ) in states


def test_current_language_always_has_positive_buffer_count() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=4,
        storage_capacity=0,
    )

    for state in state_space.states():

        if state.current_language == "":
            assert (
                state.admission_buffer_ger_count
                + state.admission_buffer_fr_count
                + state.admission_buffer_rus_count
                == 0
            )

        elif state.current_language == "GER":
            assert state.admission_buffer_ger_count > 0

        elif state.current_language == "FR":
            assert state.admission_buffer_fr_count > 0

        elif state.current_language == "RUS":
            assert state.admission_buffer_rus_count > 0

        else:
            pytest.fail(
                f"Unexpected current language: "
                f"{state.current_language!r}"
            )


def test_buffer_count_does_not_exceed_capacity() -> None:
    capacity = 4

    state_space = TabularStateSpace(
        max_buffer_capacity=capacity,
        storage_capacity=0,
    )

    for state in state_space.states():
        total = (
            state.admission_buffer_ger_count
            + state.admission_buffer_fr_count
            + state.admission_buffer_rus_count
        )

        assert total <= capacity


# ---------------------------------------------------------------------------
# Storage state enumeration
# ---------------------------------------------------------------------------


def test_storage_states_include_empty_and_non_empty_semantic_states() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=0,
        storage_capacity=2,
    )

    states = state_space.states()

    expected_storage_a_states = {
        ("", 0),
        ("GER", 1),
        ("FR", 1),
        ("RUS", 1),
        ("GER", 2),
        ("FR", 2),
        ("RUS", 2),
    }

    actual_storage_a_states = {
        (
            state.storage_a_language,
            state.storage_a_count,
        )
        for state in states
    }

    assert actual_storage_a_states == (
        expected_storage_a_states
    )


def test_empty_storage_has_empty_language() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=1,
        storage_capacity=3,
    )

    for state in state_space.states():

        if state.storage_a_count == 0:
            assert state.storage_a_language == ""

        if state.storage_b_count == 0:
            assert state.storage_b_language == ""


def test_non_empty_storage_has_valid_language() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=1,
        storage_capacity=3,
    )

    valid_languages = {
        "GER",
        "FR",
        "RUS",
    }

    for state in state_space.states():

        if state.storage_a_count > 0:
            assert state.storage_a_language in valid_languages

        if state.storage_b_count > 0:
            assert state.storage_b_language in valid_languages


def test_storage_count_does_not_exceed_capacity() -> None:
    capacity = 3

    state_space = TabularStateSpace(
        max_buffer_capacity=1,
        storage_capacity=capacity,
    )

    for state in state_space.states():
        assert 0 <= state.storage_a_count <= capacity
        assert 0 <= state.storage_b_count <= capacity


# ---------------------------------------------------------------------------
# State-space cardinality
# ---------------------------------------------------------------------------


def test_buffer_state_count_for_capacity_ten() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=10,
        storage_capacity=0,
    )

    # 1 + 3 * C(12, 3) = 661
    assert state_space.count() == 661


def test_storage_state_count_for_capacity_ten() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=0,
        storage_capacity=10,
    )

    # One empty state plus three languages for each
    # positive occupancy:
    #
    # 1 + 3 * 10 = 31
    #
    # Two independent storages:
    #
    # 31^2 = 961
    assert state_space.count() == 961


def test_full_state_space_count_for_capacity_ten() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=10,
        storage_capacity=10,
    )

    # Buffer/current-language configurations:
    #
    # 1 + 3 * C(12, 3) = 661
    #
    # Storage configurations:
    #
    # (1 + 3 * 10)^2 = 31^2 = 961
    #
    # Full semantic state space:
    #
    # 661 * 961 = 635221

    assert state_space.count() == 635221


# ---------------------------------------------------------------------------
# Configuration consistency
# ---------------------------------------------------------------------------


def test_all_states_have_consistent_storage_pairs() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=3,
        storage_capacity=3,
    )

    valid_languages = {
        "GER",
        "FR",
        "RUS",
    }

    for state in state_space.states():

        if state.storage_a_count == 0:
            assert state.storage_a_language == ""
        else:
            assert state.storage_a_language in valid_languages

        if state.storage_b_count == 0:
            assert state.storage_b_language == ""
        else:
            assert state.storage_b_language in valid_languages


def test_all_states_are_tabular_states() -> None:
    state_space = TabularStateSpace(
        max_buffer_capacity=2,
        storage_capacity=2,
    )

    assert all(
        isinstance(state, TabularState)
        for state in state_space.states()
    )
