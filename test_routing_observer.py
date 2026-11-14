from __future__ import annotations

import pytest

from routing.routing_observer import (
    LanguageShares,
    RoutingObservation,
    RoutingObserver,
    RoutingObserverSettings,
)


def test_language_shares_require_valid_distribution() -> None:
    shares = LanguageShares(
        fr=0.5,
        ger=0.2,
        rus=0.3,
    )

    assert shares.fr == 0.5
    assert shares.ger == 0.2
    assert shares.rus == 0.3


def test_language_shares_reject_negative_values() -> None:
    with pytest.raises(ValueError):
        LanguageShares(
            fr=0.5,
            ger=-0.1,
            rus=0.6,
        )


def test_language_shares_reject_invalid_sum() -> None:
    with pytest.raises(ValueError):
        LanguageShares(
            fr=0.5,
            ger=0.5,
            rus=0.5,
        )


def test_routing_observer_counts_and_computes_statistics() -> None:
    observer = RoutingObserver()

    for _ in range(5):
        observer.observe("FR")

    for _ in range(2):
        observer.observe("GER")

    for _ in range(3):
        observation = observer.observe("RUS")

    assert observation.total == 10

    assert observation.fr_count == 5
    assert observation.ger_count == 2
    assert observation.rus_count == 3

    assert observation.fr_share == pytest.approx(0.5)
    assert observation.ger_share == pytest.approx(0.2)
    assert observation.rus_share == pytest.approx(0.3)

    assert observation.fr_deviation == pytest.approx(0.0)
    assert observation.ger_deviation == pytest.approx(0.0)
    assert observation.rus_deviation == pytest.approx(0.0)

    assert observation.fr_deficit == pytest.approx(0.0)
    assert observation.ger_deficit == pytest.approx(0.0)
    assert observation.rus_deficit == pytest.approx(0.0)

    assert observation.maximum_deviation == pytest.approx(0.0)


def test_routing_observer_does_not_switch_before_minimum_observations() -> None:
    settings = RoutingObserverSettings(
        min_observations=10,
        share_threshold=0.05,
    )

    observer = RoutingObserver(settings=settings)

    for _ in range(9):
        observation = observer.observe("RUS")

    assert observation.total == 9
    assert observation.maximum_deviation == pytest.approx(0.7)
    assert observer.should_switch(observation) is False


def test_routing_observer_switches_above_deviation_threshold() -> None:
    settings = RoutingObserverSettings(
        min_observations=10,
        share_threshold=0.05,
    )

    observer = RoutingObserver(settings=settings)

    for _ in range(10):
        observation = observer.observe("RUS")

    assert observation.total == 10
    assert observation.rus_share == pytest.approx(1.0)
    assert observation.rus_deviation == pytest.approx(0.7)
    assert observation.maximum_deviation == pytest.approx(0.7)

    assert observer.should_switch(observation) is True


def test_routing_observer_does_not_switch_at_threshold_boundary() -> None:
    settings = RoutingObserverSettings(
        min_observations=10,
        share_threshold=0.05,
        comparison_tolerance=1e-12,
    )

    observer = RoutingObserver(settings=settings)

    # 11 FR, 4 GER, 5 RUS
    #
    # FR deviation = +0.05
    # GER deviation =  0.00
    # RUS deviation = -0.05
    #
    # Floating-point evaluation may produce
    # 0.050000000000000044, but that must still count
    # as being on the threshold.

    for _ in range(11):
        observer.observe("FR")

    for _ in range(4):
        observer.observe("GER")

    for _ in range(5):
        observation = observer.observe("RUS")

    assert observation.fr_deviation == pytest.approx(0.05)
    assert observation.rus_deviation == pytest.approx(-0.05)
    assert observation.maximum_deviation == pytest.approx(0.05)

    assert observer.should_switch(observation) is False


def test_routing_observer_switches_when_deviation_exceeds_threshold() -> None:
    settings = RoutingObserverSettings(
        min_observations=10,
        share_threshold=0.05,
    )

    observer = RoutingObserver(settings=settings)

    # 12 FR, 3 GER, 5 RUS
    #
    # FR = 0.60 -> +0.10
    # GER = 0.15 -> -0.05
    # RUS = 0.25 -> -0.05

    for _ in range(12):
        observer.observe("FR")

    for _ in range(3):
        observer.observe("GER")

    for _ in range(5):
        observation = observer.observe("RUS")

    assert observation.maximum_deviation == pytest.approx(0.10)

    assert observer.should_switch(observation) is True


def test_non_rus_languages_are_part_of_switching_decision() -> None:
    settings = RoutingObserverSettings(
        min_observations=10,
        share_threshold=0.05,
    )

    observer = RoutingObserver(settings=settings)

    for _ in range(5):
        observer.observe("FR")

    for _ in range(5):
        observation = observer.observe("GER")

    assert observation.fr_count == 5
    assert observation.ger_count == 5
    assert observation.rus_count == 0

    assert observation.fr_share == pytest.approx(0.5)
    assert observation.ger_share == pytest.approx(0.5)
    assert observation.rus_share == pytest.approx(0.0)

    assert observation.maximum_deviation == pytest.approx(0.3)

    assert observer.should_switch(observation) is True


def test_routing_observer_reset_clears_statistics() -> None:
    observer = RoutingObserver()

    observer.observe("FR")
    observer.observe("GER")
    observer.observe("RUS")

    observer.reset()

    observation = observer.statistics()

    assert observer.total == 0

    assert observation.total == 0
    assert observation.fr_count == 0
    assert observation.ger_count == 0
    assert observation.rus_count == 0

    assert observation.maximum_deviation == 0.0
    assert observer.should_switch(observation) is False


def test_routing_observer_rejects_unknown_language() -> None:
    observer = RoutingObserver()

    with pytest.raises(ValueError, match="Unknown language"):
        observer.observe("ENG")


def test_routing_observer_settings_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        RoutingObserverSettings(min_observations=0)

    with pytest.raises(ValueError):
        RoutingObserverSettings(share_threshold=-0.01)

    with pytest.raises(ValueError):
        RoutingObserverSettings(comparison_tolerance=-1e-12)