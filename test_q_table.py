from problem.q_table import QTable
from problem.tabular_state import TabularState


def make_state() -> TabularState:
    return TabularState(
        current_language="GER",
        admission_buffer_ger_count=1,
        admission_buffer_fr_count=1,
        admission_buffer_rus_count=0,
        storage_a_language="GER",
        storage_a_count=2,
        storage_b_language="FR",
        storage_b_count=1,
    )


def test_unknown_state_action_returns_default_value():
    table = QTable()

    state = make_state()

    assert table.get(state, 0) == 0.0


def test_custom_default_value_is_returned():
    table = QTable(default_value=-1.5)

    state = make_state()

    assert table.get(state, 2) == -1.5


def test_set_and_get_value():
    table = QTable()

    state = make_state()

    table.set(state, 1, 3.25)

    assert table.get(state, 1) == 3.25


def test_different_actions_have_independent_values():
    table = QTable()

    state = make_state()

    table.set(state, 0, 1.0)
    table.set(state, 1, 2.0)
    table.set(state, 2, 3.0)

    assert table.get(state, 0) == 1.0
    assert table.get(state, 1) == 2.0
    assert table.get(state, 2) == 3.0


def test_values_returns_values_in_action_order():
    table = QTable()

    state = make_state()

    table.set(state, 0, 1.0)
    table.set(state, 1, 2.0)
    table.set(state, 2, 3.0)

    assert table.values(
        state,
        (2, 0, 1),
    ) == (
        3.0,
        1.0,
        2.0,
    )


def test_different_states_have_independent_values():
    table = QTable()

    state_a = TabularState(
        current_language="GER",
        admission_buffer_ger_count=1,
        admission_buffer_fr_count=1,
        admission_buffer_rus_count=0,
        storage_a_language="GER",
        storage_a_count=2,
        storage_b_language="FR",
        storage_b_count=1,
    )

    state_b = TabularState(
        current_language="FR",
        admission_buffer_ger_count=0,
        admission_buffer_fr_count=2,
        admission_buffer_rus_count=0,
        storage_a_language="GER",
        storage_a_count=1,
        storage_b_language="FR",
        storage_b_count=2,
    )

    table.set(state_a, 0, 5.0)

    assert table.get(state_a, 0) == 5.0
    assert table.get(state_b, 0) == 0.0

def test_snapshot_returns_all_q_values():
    table = QTable()

    state = make_state()

    table.set(state, 0, 1.25)
    table.set(state, 2, -3.5)

    assert table.snapshot() == {
        (state, 0): 1.25,
        (state, 2): -3.5,
    }


def test_snapshot_is_independent_from_q_table():
    table = QTable()

    state = make_state()

    table.set(state, 0, 1.25)

    snapshot = table.snapshot()

    table.set(state, 0, 9.0)

    assert snapshot[(state, 0)] == 1.25
    assert table.get(state, 0) == 9.0


def test_restore_reconstructs_q_values():
    table = QTable()

    state = make_state()

    table.restore(
        {
            (state, 0): 2.5,
            (state, 1): -1.0,
            (state, 2): 7.25,
        }
    )

    assert table.get(state, 0) == 2.5
    assert table.get(state, 1) == -1.0
    assert table.get(state, 2) == 7.25


def test_restore_replaces_existing_q_values():
    table = QTable()

    state = make_state()

    table.set(state, 0, 10.0)
    table.set(state, 1, 20.0)

    table.restore(
        {
            (state, 2): 5.0,
        }
    )

    assert table.get(state, 0) == 0.0
    assert table.get(state, 1) == 0.0
    assert table.get(state, 2) == 5.0


def test_restore_normalizes_values_to_float():
    table = QTable()

    state = make_state()

    table.restore(
        {
            (state, 0): 3,
        }
    )

    assert table.get(state, 0) == 3.0
    assert isinstance(
        table.get(state, 0),
        float,
    )