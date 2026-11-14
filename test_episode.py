from __future__ import annotations

from environment.episode import (
    EpisodeOutcome,
    EpisodeResult,
)

import pytest

from environment.termination import TerminationReason
from environment.episode import (
    EpisodeOutcome,
    EpisodeResult,
    episode_outcome_from_termination_reason,
)

def test_episode_result_success() -> None:
    result = EpisodeResult(
        episode_number=1,
        outcome=EpisodeOutcome.SUCCESS,
        decision_points=100,
        final_decision_point_id=100,
    )

    assert result.successful is True
    assert result.failed is False


def test_episode_result_failure() -> None:
    result = EpisodeResult(
        episode_number=2,
        outcome=EpisodeOutcome.FAILURE,
        decision_points=50,
        final_decision_point_id=50,
        reason="Manufacturing failure",
    )

    assert result.successful is False
    assert result.failed is True


def test_episode_result_communication_failure() -> None:
    result = EpisodeResult(
        episode_number=3,
        outcome=EpisodeOutcome.COMMUNICATION_FAILURE,
        decision_points=42,
        final_decision_point_id=42,
        reason="TCP connection closed.",
    )

    assert result.successful is False
    assert result.failed is True


def test_episode_result_interrupted() -> None:
    result = EpisodeResult(
        episode_number=4,
        outcome=EpisodeOutcome.INTERRUPTED,
        decision_points=27,
        final_decision_point_id=27,
        reason="User interruption.",
    )

    assert result.successful is False
    assert result.failed is False





def test_episode_result_success() -> None:
    result = EpisodeResult(
        episode_number=1,
        outcome=EpisodeOutcome.SUCCESS,
        decision_points=25,
        final_decision_point_id=25,
    )

    assert result.successful is True
    assert result.failed is False
    assert result.interrupted is False


@pytest.mark.parametrize(
    "outcome",
    [
        EpisodeOutcome.MANUFACTURING_FAILURE,
        EpisodeOutcome.HORIZON_TIMEOUT,
        EpisodeOutcome.COMMUNICATION_FAILURE,
    ],
)
def test_episode_result_failure(
    outcome: EpisodeOutcome,
) -> None:
    result = EpisodeResult(
        episode_number=1,
        outcome=outcome,
        decision_points=25,
    )

    assert result.successful is False
    assert result.failed is True
    assert result.interrupted is False


def test_episode_result_interrupted() -> None:
    result = EpisodeResult(
        episode_number=1,
        outcome=EpisodeOutcome.INTERRUPTED,
        decision_points=10,
        reason="KeyboardInterrupt",
    )

    assert result.successful is False
    assert result.failed is False
    assert result.interrupted is True


@pytest.mark.parametrize(
    ("termination_reason", "expected_outcome"),
    [
        (
            TerminationReason.SUCCESS,
            EpisodeOutcome.SUCCESS,
        ),
        (
            TerminationReason.MANUFACTURING_FAILURE,
            EpisodeOutcome.MANUFACTURING_FAILURE,
        ),
        (
            TerminationReason.HORIZON_TIMEOUT,
            EpisodeOutcome.HORIZON_TIMEOUT,
        ),
        (
            TerminationReason.COMMUNICATION_FAILURE,
            EpisodeOutcome.COMMUNICATION_FAILURE,
        ),
    ],
)
def test_episode_outcome_maps_termination_reason(
    termination_reason: TerminationReason,
    expected_outcome: EpisodeOutcome,
) -> None:
    assert (
        episode_outcome_from_termination_reason(
            termination_reason
        )
        is expected_outcome
    )