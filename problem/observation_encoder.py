from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from problem.observation import Observation


@dataclass(frozen=True)
class ObservationEncoderSettings:
    """Configuration of the numerical observation representation."""

    storage_a_normalization_capacity: int
    storage_b_normalization_capacity: int

    return_buffer_capacity: int
    target_throughput: int


class ObservationEncoder:
    """
    Convert a semantic observation into a normalized float32 tensor.

    The resulting tensor is intended for consumption by the agent.
    """

    LANGUAGE_DIM = 4

    EMPTY_LANGUAGE = ""

    LANGUAGE_ENCODING: dict[str, int] = {
        "GER": 0,
        "FR": 1,
        "RUS": 2,
        "": 3,
    }

    def __init__(
        self,
        settings: ObservationEncoderSettings,
    ) -> None:
        self.settings = settings

        self.dimension = (
            self.LANGUAGE_DIM
            + 1
            + self.LANGUAGE_DIM
            + 1
            + self.LANGUAGE_DIM
            + 1
            + 1
        )

    def encode(
        self,
        observation: Observation,
    ) -> Tensor:
        features: list[float] = []

        features.append(
            self._normalize_count(
                observation.storage_a_count,
                self.settings
                .storage_a_normalization_capacity,
            )
        )

        features.extend(
            self._one_hot_language(
                observation.storage_a_language
            )
        )

        features.append(
            self._normalize_count(
                observation.storage_b_count,
                self.settings
                .storage_b_normalization_capacity,
            )
        )

        features.extend(
            self._one_hot_language(
                observation.storage_b_language
            )
        )

        normalized_throughput = (
            observation.throughput
            / self.settings.target_throughput
        )

        normalized_throughput = min(
            max(normalized_throughput, 0.0),
            1.0,
        )

        features.append(normalized_throughput)

        return torch.tensor(
            features,
            dtype=torch.float32,
        )

    @classmethod
    def _one_hot_language(
        cls,
        language: str,
    ) -> list[float]:
        if language not in cls.LANGUAGE_ENCODING:
            raise ValueError(
                f"Unknown language: {language!r}. "
                f"Expected one of "
                f"{list(cls.LANGUAGE_ENCODING)}."
            )

        index = cls.LANGUAGE_ENCODING[language]

        return cls._one_hot(
            index,
            cls.LANGUAGE_DIM,
        )

    @staticmethod
    def _one_hot(
        index: int,
        dimension: int,
    ) -> list[float]:
        if not 0 <= index < dimension:
            raise ValueError(
                f"Index {index} outside "
                f"[0, {dimension - 1}]."
            )

        result = [0.0] * dimension
        result[index] = 1.0

        return result

    @staticmethod
    def _normalize_count(
        count: int,
        normalization_capacity: int,
    ) -> float:
        if not isinstance(count, int):
            raise TypeError(
                f"Count must be int, "
                f"got {type(count).__name__}."
            )

        if normalization_capacity <= 0:
            raise ValueError(
                "Normalization capacity must "
                "be greater than zero."
            )

        if count < 0:
            raise ValueError(
                "Count cannot be negative."
            )

        if count > normalization_capacity:
            raise ValueError(
                f"Count {count} exceeds "
                f"normalization capacity "
                f"{normalization_capacity}."
            )

        return (
            float(count)
            / float(normalization_capacity)
        )