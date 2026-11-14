from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageShares:
    """Target production shares for the three languages."""

    fr: float = 0.5
    ger: float = 0.2
    rus: float = 0.3

    def __post_init__(self) -> None:
        shares = (
            self.fr,
            self.ger,
            self.rus,
        )

        if any(share < 0.0 for share in shares):
            raise ValueError(
                "Language shares must be non-negative."
            )

        total = sum(shares)

        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "Language shares must sum to 1.0. "
                f"Got {total}."
            )


@dataclass(frozen=True)
class RoutingObserverSettings:
    """Configuration of cumulative language observation."""

    target_shares: LanguageShares = LanguageShares()

    # Minimum number of observations before switching
    # can be considered.
    min_observations: int = 200

    # Maximum permitted absolute deviation from the
    # target language distribution.
    share_threshold: float = 0.05

    # Numerical tolerance used when comparing deviations
    # against the switching threshold.
    comparison_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.min_observations < 1:
            raise ValueError(
                "min_observations must be positive."
            )

        if self.share_threshold < 0.0:
            raise ValueError(
                "share_threshold must be non-negative."
            )

        if self.comparison_tolerance < 0.0:
            raise ValueError(
                "comparison_tolerance must be non-negative."
            )


@dataclass(frozen=True)
class RoutingObservation:
    """Current cumulative language statistics."""

    total: int

    fr_count: int
    ger_count: int
    rus_count: int

    fr_share: float
    ger_share: float
    rus_share: float

    fr_deviation: float
    ger_deviation: float
    rus_deviation: float

    fr_deficit: float
    ger_deficit: float
    rus_deficit: float

    @property
    def maximum_deviation(self) -> float:
        """Maximum absolute deviation from the target distribution."""

        return max(
            abs(self.fr_deviation),
            abs(self.ger_deviation),
            abs(self.rus_deviation),
        )


@dataclass
class RoutingObserver:
    """
    Observe cumulative source language frequencies.

    The observer maintains production statistics and evaluates
    whether the observed language distribution has deviated
    sufficiently from the configured target distribution.

    The observer does not select a routing rule.
    """

    settings: RoutingObserverSettings = RoutingObserverSettings()

    fr_count: int = 0
    ger_count: int = 0
    rus_count: int = 0

    def reset(self) -> None:
        """Reset cumulative language counts."""

        self.fr_count = 0
        self.ger_count = 0
        self.rus_count = 0

    def observe(
        self,
        language: str,
    ) -> RoutingObservation:
        """Observe one incoming palette."""

        if language == "FR":
            self.fr_count += 1

        elif language == "GER":
            self.ger_count += 1

        elif language == "RUS":
            self.rus_count += 1

        else:
            raise ValueError(
                f"Unknown language {language!r}. "
                "Expected one of {'FR', 'GER', 'RUS'}."
            )

        return self.statistics()

    def statistics(self) -> RoutingObservation:
        """Return the current cumulative statistics."""

        total = self.total

        if total == 0:
            return RoutingObservation(
                total=0,

                fr_count=0,
                ger_count=0,
                rus_count=0,

                fr_share=0.0,
                ger_share=0.0,
                rus_share=0.0,

                fr_deviation=0.0,
                ger_deviation=0.0,
                rus_deviation=0.0,

                fr_deficit=0.0,
                ger_deficit=0.0,
                rus_deficit=0.0,
            )

        shares = self.settings.target_shares

        fr_share = self.fr_count / total
        ger_share = self.ger_count / total
        rus_share = self.rus_count / total

        fr_deviation = fr_share - shares.fr
        ger_deviation = ger_share - shares.ger
        rus_deviation = rus_share - shares.rus

        fr_deficit = (
            self.fr_count
            - shares.fr * total
        )

        ger_deficit = (
            self.ger_count
            - shares.ger * total
        )

        rus_deficit = (
            self.rus_count
            - shares.rus * total
        )

        return RoutingObservation(
            total=total,

            fr_count=self.fr_count,
            ger_count=self.ger_count,
            rus_count=self.rus_count,

            fr_share=fr_share,
            ger_share=ger_share,
            rus_share=rus_share,

            fr_deviation=fr_deviation,
            ger_deviation=ger_deviation,
            rus_deviation=rus_deviation,

            fr_deficit=fr_deficit,
            ger_deficit=ger_deficit,
            rus_deficit=rus_deficit,
        )

    def should_switch(
        self,
        observation: RoutingObservation | None = None,
    ) -> bool:
        """
        Determine whether the routing regime should be reconsidered.

        Switching requires:

            total >= min_observations

        and

            maximum_deviation > share_threshold

        where comparison_tolerance protects the decision from
        floating-point round-off at the threshold boundary.
        """

        if observation is None:
            observation = self.statistics()

        if observation.total < self.settings.min_observations:
            return False

        threshold = (
            self.settings.share_threshold
            + self.settings.comparison_tolerance
        )

        return observation.maximum_deviation > threshold

    @property
    def total(self) -> int:
        """Number of observed palettes."""

        return (
            self.fr_count
            + self.ger_count
            + self.rus_count
        )