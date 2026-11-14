from problem.reachable_state_registry import (
    ReachableStateRegistry,
)
from problem.tabular_state import TabularState


def make_state(
    current_language: str = "GER",
    ger_count: int = 1,
    fr_count: int = 0,
    rus_count: int = 0,
    storage_a_language: str = "",
    storage_b_language: str = "",
) -> TabularState:
    return TabularState(
        current_language=current_language,
        admission_buffer_ger_count=ger_count,
        admission_buffer_fr_count=fr_count,
        admission_buffer_rus_count=rus_count,
        storage_a_language=storage_a_language,
        storage_a_count=0,
        storage_b_language=storage_b_language,
        storage_b_count=0,
    )


def test_empty_registry_contains_no_states():
    registry = ReachableStateRegistry()

    state = make_state()

    assert registry.count() == 0
    assert not registry.contains(state)
    assert registry.visits(state) == 0
    assert registry.states() == ()


def test_observe_registers_new_state():
    registry = ReachableStateRegistry()

    state = make_state()

    registry.observe(state)

    assert registry.count() == 1
    assert registry.contains(state)
    assert registry.visits(state) == 1


def test_observing_same_state_increments_visits():
    registry = ReachableStateRegistry()

    state = make_state()

    registry.observe(state)
    registry.observe(state)
    registry.observe(state)

    assert registry.count() == 1
    assert registry.visits(state) == 3


def test_different_states_are_registered_independently():
    registry = ReachableStateRegistry()

    state_a = make_state(
        current_language="GER",
    )

    state_b = make_state(
        current_language="FR",
        ger_count=0,
        fr_count=1,
    )

    registry.observe(state_a)
    registry.observe(state_b)

    assert registry.count() == 2
    assert registry.visits(state_a) == 1
    assert registry.visits(state_b) == 1


def test_unknown_state_has_zero_visits():
    registry = ReachableStateRegistry()

    known_state = make_state()
    unknown_state = make_state(
        current_language="RUS",
        ger_count=0,
        rus_count=1,
    )

    registry.observe(known_state)

    assert registry.visits(unknown_state) == 0
    assert not registry.contains(unknown_state)


def test_states_returns_registered_states():
    registry = ReachableStateRegistry()

    state_a = make_state()

    state_b = make_state(
        current_language="FR",
        ger_count=0,
        fr_count=1,
    )

    registry.observe(state_a)
    registry.observe(state_b)
    registry.observe(state_a)

    assert registry.states() == (
        state_a,
        state_b,
    )


def test_state_order_follows_first_observation():
    registry = ReachableStateRegistry()

    state_a = make_state(
        current_language="GER",
    )

    state_b = make_state(
        current_language="RUS",
        ger_count=0,
        rus_count=1,
    )

    state_c = make_state(
        current_language="FR",
        ger_count=0,
        fr_count=1,
    )

    registry.observe(state_b)
    registry.observe(state_a)
    registry.observe(state_c)
    registry.observe(state_b)

    assert registry.states() == (
        state_b,
        state_a,
        state_c,
    )