import pytest

from problem.observation import Observation
from problem.tabular_state import TabularState
from problem.tabular_state_encoder import TabularStateEncoder
from problem.problem import ManufacturingProblem

def make_observation(
    *,
    admission_buffer_contents=(
        "GER",
        "FR",
        "RUS",
        "GER",
    ),
    storage_a_language="GER",
    storage_b_language="FR",
) -> Observation:
    return Observation(
        simulation_time=10.0,
        admission_buffer_contents=admission_buffer_contents,
        storage_a_count=4,
        storage_a_language=storage_a_language,
        storage_b_count=3,
        storage_b_language=storage_b_language,
        throughput=20,
    )

def make_encoder(
    *,
    admission_buffer_capacity: int = 10,
) -> TabularStateEncoder:
    problem = ManufacturingProblem(
        admission_buffer_capacity=admission_buffer_capacity,
    )

    return TabularStateEncoder(problem)

def test_encoder_extracts_current_language_from_admission_buffer():
    encoder = make_encoder()

    state = encoder.encode(
        make_observation(
            admission_buffer_contents=(
                "RUS",
                "GER",
                "FR",
            )
        )
    )

    assert state.current_language == "RUS"


def test_encoder_counts_each_buffer_language():
    encoder = make_encoder()

    state = encoder.encode(
        make_observation(
            admission_buffer_contents=(
                "GER",
                "FR",
                "RUS",
                "GER",
                "GER",
                "FR",
            )
        )
    )

    assert state.admission_buffer_ger_count == 3
    assert state.admission_buffer_fr_count == 2
    assert state.admission_buffer_rus_count == 1


def test_buffer_permutation_preserves_composition_counts():
    encoder = make_encoder()

    state_a = encoder.encode(
        make_observation(
            admission_buffer_contents=(
                "GER",
                "FR",
                "RUS",
                "GER",
            )
        )
    )

    state_b = encoder.encode(
        make_observation(
            admission_buffer_contents=(
                "RUS",
                "GER",
                "GER",
                "FR",
            )
        )
    )

    assert (
        state_a.admission_buffer_ger_count
        == state_b.admission_buffer_ger_count
    )
    assert (
        state_a.admission_buffer_fr_count
        == state_b.admission_buffer_fr_count
    )
    assert (
        state_a.admission_buffer_rus_count
        == state_b.admission_buffer_rus_count
    )


def test_current_language_is_still_preserved_when_composition_is_equal():
    encoder = make_encoder()

    state_a = encoder.encode(
        make_observation(
            admission_buffer_contents=(
                "GER",
                "FR",
                "RUS",
            )
        )
    )

    state_b = encoder.encode(
        make_observation(
            admission_buffer_contents=(
                "FR",
                "GER",
                "RUS",
            )
        )
    )

    assert state_a != state_b

    assert state_a.current_language == "GER"
    assert state_b.current_language == "FR"

    assert (
        state_a.admission_buffer_ger_count
        == state_b.admission_buffer_ger_count
    )
    assert (
        state_a.admission_buffer_fr_count
        == state_b.admission_buffer_fr_count
    )
    assert (
        state_a.admission_buffer_rus_count
        == state_b.admission_buffer_rus_count
    )


def test_empty_admission_buffer_has_zero_counts():
    encoder = make_encoder()

    state = encoder.encode(
        make_observation(
            admission_buffer_contents=()
        )
    )

    assert state == TabularState(
        current_language="",
        admission_buffer_ger_count=0,
        admission_buffer_fr_count=0,
        admission_buffer_rus_count=0,
        storage_a_language="GER",
        storage_a_count=4,
        storage_b_language="FR",
        storage_b_count=3,
    )


def test_storage_languages_are_preserved():
    encoder = make_encoder()

    state = encoder.encode(
        make_observation(
            storage_a_language="RUS",
            storage_b_language="",
        )
    )

    assert state.storage_a_language == "RUS"
    assert state.storage_b_language == ""


def test_unknown_buffer_language_is_rejected():
    encoder = make_encoder()

    observation = make_observation(
        admission_buffer_contents=(
            "GER",
            "ESP",
            "RUS",
        )
    )

    with pytest.raises(ValueError):
        encoder.encode(observation)


def test_unknown_storage_language_is_rejected():
    encoder = make_encoder()

    observation = make_observation(
        storage_a_language="ESP"
    )

    with pytest.raises(ValueError):
        encoder.encode(observation)


def test_storage_occupancies_are_preserved() -> None:
    observation = Observation(
        simulation_time=10.0,
        admission_buffer_contents=(
            "GER",
            "FR",
        ),
        storage_a_count=7,
        storage_a_language="GER",
        storage_b_count=3,
        storage_b_language="FR",
        throughput=20,
    )

    encoder = make_encoder()

    state = encoder.encode(observation)

    assert state.storage_a_language == "GER"
    assert state.storage_a_count == 7

    assert state.storage_b_language == "FR"
    assert state.storage_b_count == 3


def test_different_storage_occupancies_produce_different_states() -> None:
    encoder = make_encoder()

    observation_a = Observation(
        simulation_time=10.0,
        admission_buffer_contents=("GER",),
        storage_a_count=2,
        storage_a_language="GER",
        storage_b_count=3,
        storage_b_language="FR",
        throughput=20,
    )

    observation_b = Observation(
        simulation_time=10.0,
        admission_buffer_contents=("GER",),
        storage_a_count=10,
        storage_a_language="GER",
        storage_b_count=3,
        storage_b_language="FR",
        throughput=20,
    )

    state_a = encoder.encode(observation_a)
    state_b = encoder.encode(observation_b)

    assert state_a != state_b