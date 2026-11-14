from __future__ import annotations

from abc import ABC, abstractmethod

from problem.observation import Observation


class ObservationSource(ABC):
    """Abstract source of semantic RL observations."""

    @abstractmethod
    def read(self) -> Observation:
        """Read the current semantic observation."""
        raise NotImplementedError