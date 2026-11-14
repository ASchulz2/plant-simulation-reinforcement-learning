from __future__ import annotations

from problem.observation_source import ObservationSource
from problem.observation import Observation


class PlantObservationReader(ObservationSource):

    ADMISSION_BUFFER_PATH = (
        ".Models.Model.AdmissionBuffer"
    )

    def __init__(self, plant) -> None:
        self.plant = plant

    def read(self) -> Observation:
        return Observation(
            simulation_time=float(
                self.plant.get_value(
                    ".Models.Model.EventController.SimTime"
                )
            ),

            admission_buffer_contents=(
                self._read_admission_buffer_contents()
            ),

            storage_a_count=int(
                self.plant.get_value(
                    ".Models.Model.StorageA.NumMU"
                )
            ),

            storage_a_language=str(
                self.plant.get_value(
                    ".Models.Model.StorageA.storedLang"
                )
            ),

            storage_b_count=int(
                self.plant.get_value(
                    ".Models.Model.StorageB.NumMU"
                )
            ),

            storage_b_language=str(
                self.plant.get_value(
                    ".Models.Model.StorageB.storedLang"
                )
            ),

            throughput=(
                int(
                    self.plant.get_value(
                        ".Models.Model.DrainA.NumPartsSinceSetup"
                    )
                )
                +
                int(
                    self.plant.get_value(
                        ".Models.Model.DrainB.NumPartsSinceSetup"
                    )
                )
            ),
        )

    def _read_admission_buffer_contents(
        self,
    ) -> tuple[str, ...]:
        """
        Read the cached AdmissionBuffer language representation.

        Plant Simulation maintains the scalar
        RL_AdmissionBufferLanguages attribute. Python deliberately
        receives only plain string data across COM.
        """

        raw = self.plant.get_value(
            ".Models.Model.AdmissionBuffer."
            "RL_AdmissionBufferLanguages"
        )

        text = str(raw)

        if not text:
            return ()

        return tuple(
            language
            for language in text.split("|")
            if language
        )