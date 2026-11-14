from __future__ import annotations

from problem.observation import Observation
from problem.observation_source import ObservationSource
from environment.plant_observation_reader import (
    PlantObservationReader,
)


class FakePlant:
    """Minimal Plant Simulation stand-in for observation tests."""

    def __init__(self) -> None:
        self.values = {
            ".Models.Model.EventController.SimTime": 12.5,

            ".Models.Model.AdmissionBuffer.RL_AdmissionBufferLanguages":
                "GER|FR|GER",

            ".Models.Model.StorageA.NumMU": 3,
            ".Models.Model.StorageA.storedLang": "GER",

            ".Models.Model.StorageB.NumMU": 2,
            ".Models.Model.StorageB.storedLang": "FR",

            ".Models.Model.DrainA.NumPartsSinceSetup": 7,
            ".Models.Model.DrainB.NumPartsSinceSetup": 3,
        }

    def get_value(
        self,
        path: str,
    ):
        return self.values[path]


def test_plant_observation_reader_is_observation_source() -> None:
    plant = FakePlant()

    reader = PlantObservationReader(plant)

    assert isinstance(
        reader,
        ObservationSource,
    )


def test_read_returns_semantic_observation() -> None:
    plant = FakePlant()

    reader = PlantObservationReader(plant)

    observation = reader.read()

    assert isinstance(
        observation,
        Observation,
    )

    assert observation.simulation_time == 12.5

    assert observation.admission_buffer_contents == (
        "GER",
        "FR",
        "GER",
    )

    assert observation.storage_a_count == 3
    assert observation.storage_a_language == "GER"

    assert observation.storage_b_count == 2
    assert observation.storage_b_language == "FR"

    assert observation.throughput == 10


def test_empty_admission_buffer() -> None:
    plant = FakePlant()

    plant.values[
        ".Models.Model.AdmissionBuffer.RL_AdmissionBufferLanguages"
    ] = ""

    reader = PlantObservationReader(plant)

    observation = reader.read()

    assert observation.admission_buffer_contents == ()


def test_admission_buffer_order_is_preserved() -> None:
    plant = FakePlant()

    plant.values[
        ".Models.Model.AdmissionBuffer.RL_AdmissionBufferLanguages"
    ] = "RUS|GER|FR"

    reader = PlantObservationReader(plant)

    observation = reader.read()

    assert observation.admission_buffer_contents == (
        "RUS",
        "GER",
        "FR",
    )
