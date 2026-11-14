from problem.tabular_state import TabularState

def test_tabular_state_contains_storage_occupancies() -> None:
    state = TabularState(
        current_language="GER",
        admission_buffer_ger_count=4,
        admission_buffer_fr_count=2,
        admission_buffer_rus_count=1,
        storage_a_language="GER",
        storage_a_count=7,
        storage_b_language="FR",
        storage_b_count=3,
    )

    assert state.storage_a_language == "GER"
    assert state.storage_a_count == 7

    assert state.storage_b_language == "FR"
    assert state.storage_b_count == 3