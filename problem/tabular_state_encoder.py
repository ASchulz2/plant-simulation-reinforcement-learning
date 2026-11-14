from __future__ import annotations

from collections import Counter

from problem.observation import Observation
from problem.tabular_state import TabularState
from problem.problem import ManufacturingProblem


class TabularStateEncoder:
    """
    Convert a semantic Observation into a discrete tabular state.

    The first AdmissionBuffer item defines the current MU.
    The complete buffer is additionally reduced to language counts.
    """

    EMPTY_LANGUAGE = ""

    VALID_LANGUAGES = {
        "GER",
        "FR",
        "RUS",
        "",
    }

    def __init__(
            self,
            problem: ManufacturingProblem,
            *,
            language_only: bool = False
        ) -> None:
        self._problem = problem
        self._language_only = language_only

    def encode(
        self,
        observation: Observation,
    ) -> TabularState:
        current_language = self._current_language(
            observation
        )

        self._validate_language(
            current_language,
            "current language",
        )

        self._validate_language(
            observation.storage_a_language,
            "Storage A language",
        )

        self._validate_language(
            observation.storage_b_language,
            "Storage B language",
        )

        buffer_counts = self._buffer_counts(
            observation.admission_buffer_contents
        )

        return TabularState(
            current_language=current_language,
            admission_buffer_ger_count=(
                buffer_counts["GER"]
            ),
            admission_buffer_fr_count=(
                buffer_counts["FR"]
            ),
            admission_buffer_rus_count=(
                buffer_counts["RUS"]
            ),
            storage_a_language=(
                observation.storage_a_language
            ),
            storage_a_count=(
                observation.storage_a_count
            ),
            storage_b_language=(
                observation.storage_b_language
            ),
            storage_b_count=(
                observation.storage_b_count
            ),
        )

    def _current_language(
        self,
        observation: Observation,
    ) -> str:
        if not observation.admission_buffer_contents:
            return self.EMPTY_LANGUAGE

        return observation.admission_buffer_contents[0]

    def _buffer_counts(
        self,
        contents: tuple[str, ...],
    ) -> dict[str, int]:

        if len(contents) > self._problem.admission_buffer_capacity:
            raise ValueError(
                "AdmissionBuffer exceeds the configured capacity."
            )

        for language in contents:
            self._validate_language(
                language,
                "AdmissionBuffer language",
            )

        counts = Counter(contents)

        return {
            "GER": counts.get("GER", 0),
            "FR": counts.get("FR", 0),
            "RUS": counts.get("RUS", 0),
        }

    @classmethod
    def _validate_language(
        cls,
        language: str,
        name: str,
    ) -> None:
        if language not in cls.VALID_LANGUAGES:
            raise ValueError(
                f"Unknown {name}: {language!r}. "
                f"Expected one of "
                f"{sorted(cls.VALID_LANGUAGES)!r}."
            )