from problem.state_action_registry import (
    StateActionVisitRegistry,
)
from problem.tabular_state import TabularState


def make_state(
    current_language: str = "GER",
) -> TabularState:
    if current_language == "GER":
        return TabularState(
            current_language="GER",
            admission_buffer_ger_count=1,
            admission_buffer_fr_count=0,
            admission_buffer_rus_count=0,
            storage_a_language="",
            storage_a_count=0,
            storage_b_language="",
            storage_b_count=0,
        )

    if current_language == "FR":
        return TabularState(
            current_language="FR",
            admission_buffer_ger_count=1,
            admission_buffer_fr_count=0,
            admission_buffer_rus_count=0,
            storage_a_language="",
            storage_a_count=0,
            storage_b_language="",
            storage_b_count=0,
        )

    return TabularState(
        current_language="RUS",
        admission_buffer_ger_count=1,
        admission_buffer_fr_count=0,
        admission_buffer_rus_count=0,
        storage_a_language="",
        storage_a_count=0,
        storage_b_language="",
        storage_b_count=0,
    )


def test_empty_registry_contains_no_state_action_pairs():
    registry = StateActionVisitRegistry()

    state = make_state()

    assert registry.count() == 0
    assert not registry.contains(state, 0)
    assert registry.visits(state, 0) == 0
    assert registry.state_actions() == ()


def test_observe_registers_state_action_pair():
    registry = StateActionVisitRegistry()

    state = make_state()

    registry.observe(state, 1)

    assert registry.count() == 1
    assert registry.contains(state, 1)
    assert registry.visits(state, 1) == 1


def test_observing_same_state_action_increments_visits():
    registry = StateActionVisitRegistry()

    state = make_state()

    registry.observe(state, 1)
    registry.observe(state, 1)
    registry.observe(state, 1)

    assert registry.count() == 1
    assert registry.visits(state, 1) == 3


def test_different_actions_are_tracked_independently():
    registry = StateActionVisitRegistry()

    state = make_state()

    registry.observe(state, 0)
    registry.observe(state, 1)
    registry.observe(state, 1)

    assert registry.count() == 2
    assert registry.visits(state, 0) == 1
    assert registry.visits(state, 1) == 2


def test_different_states_are_tracked_independently():
    registry = StateActionVisitRegistry()

    state_a = make_state("GER")
    state_b = make_state("FR")

    registry.observe(state_a, 0)
    registry.observe(state_b, 0)

    assert registry.count() == 2
    assert registry.visits(state_a, 0) == 1
    assert registry.visits(state_b, 0) == 1


def test_unknown_state_action_has_zero_visits():
    registry = StateActionVisitRegistry()

    state = make_state()

    registry.observe(state, 0)

    assert registry.visits(state, 1) == 0
    assert not registry.contains(state, 1)


def test_state_actions_preserve_first_observation_order():
    registry = StateActionVisitRegistry()

    state_a = make_state("GER")
    state_b = make_state("FR")

    registry.observe(state_b, 2)
    registry.observe(state_a, 0)
    registry.observe(state_b, 1)
    registry.observe(state_b, 2)

    assert registry.state_actions() == (
        (state_b, 2),
        (state_a, 0),
        (state_b, 1),
    )