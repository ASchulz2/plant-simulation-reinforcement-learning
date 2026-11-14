from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """
    Semantic warehouse state presented to the RL agent.

    This is deliberately independent of Plant Simulation and
    independent of the neural-network representation.
    """

    simulation_time: float

    admission_buffer_contents: tuple[str, ...]

    storage_a_count: int
    storage_a_language: str

    storage_b_count: int
    storage_b_language: str

    throughput: int