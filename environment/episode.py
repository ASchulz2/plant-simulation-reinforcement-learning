from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from environment.termination import TerminationReason


class EpisodeOutcome(str, Enum):
    """Final outcome of one experimental episode."""

    SUCCESS = "success"
    MANUFACTURING_FAILURE = "manufacturing_failure"
    HORIZON_TIMEOUT = "horizon_timeout"
    COMMUNICATION_FAILURE = "communication_failure"

    # Runner-level outcome. This is not an environment terminal reason.
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class EpisodeResult:
    """Final result of one experimental episode."""

    episode_number: int
    outcome: EpisodeOutcome

    decision_points: int
    final_decision_point_id: int | None = None

    reason: str | None = None

    @property
    def successful(self) -> bool:
        return self.outcome is EpisodeOutcome.SUCCESS

    @property
    def failed(self) -> bool:
        return self.outcome in {
            EpisodeOutcome.MANUFACTURING_FAILURE,
            EpisodeOutcome.HORIZON_TIMEOUT,
            EpisodeOutcome.COMMUNICATION_FAILURE,
        }

    @property
    def interrupted(self) -> bool:
        return self.outcome is EpisodeOutcome.INTERRUPTED


def episode_outcome_from_termination_reason(
    termination_reason: TerminationReason,
) -> EpisodeOutcome:
    """Map an environment termination reason to an episode outcome."""

    mapping = {
        TerminationReason.SUCCESS: EpisodeOutcome.SUCCESS,
        TerminationReason.MANUFACTURING_FAILURE: (
            EpisodeOutcome.MANUFACTURING_FAILURE
        ),
        TerminationReason.HORIZON_TIMEOUT: (
            EpisodeOutcome.HORIZON_TIMEOUT
        ),
        TerminationReason.COMMUNICATION_FAILURE: (
            EpisodeOutcome.COMMUNICATION_FAILURE
        ),
    }

    try:
        return mapping[termination_reason]
    except KeyError as exc:
        raise ValueError(
            "Unsupported termination reason: "
            f"{termination_reason!r}"
        ) from exc