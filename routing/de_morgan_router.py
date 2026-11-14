from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Destination(IntEnum):
    """Physical routing destinations of the Pick-and-Place Robot."""

    STORAGE_A = 0
    STORAGE_B = 1
    RETURN_BUFFER = 2


@dataclass(frozen=True)
class RoutingRule:
    """
    Mapping from product language to physical destination.
    """

    ger: Destination
    fr: Destination
    rus: Destination

    def destination(
        self,
        language: str,
    ) -> Destination:
        """Return the destination assigned to a language."""

        if language == "GER":
            return self.ger

        if language == "FR":
            return self.fr

        if language == "RUS":
            return self.rus

        raise ValueError(
            f"Unknown language {language!r}. "
            "Expected one of {'GER', 'FR', 'RUS'}."
        )

    def as_dict(
        self,
    ) -> dict[str, Destination]:
        """Return the routing rule as a dictionary."""

        return {
            "GER": self.ger,
            "FR": self.fr,
            "RUS": self.rus,
        }


@dataclass
class DeMorganRouter:
    """
    Deterministic language-to-destination router.

    The router contains no learning logic. It represents the
    current routing permutation that can later be selected by
    an observer, a De Morgan perceptron, or an RL controller.
    """

    rule: RoutingRule

    def route(
        self,
        language: str,
    ) -> Destination:
        """
        Determine the physical destination for a language.
        """

        return self.rule.destination(
            language
        )

    def set_rule(
        self,
        rule: RoutingRule,
    ) -> None:
        """Replace the active routing rule."""

        self.rule = rule

    @property
    def mapping(
        self,
    ) -> dict[str, Destination]:
        """Return the currently active routing mapping."""

        return self.rule.as_dict()

    def describe(self) -> str:
        """Return a human-readable description of the routing rule."""

        return (
            f"GER -> {self.rule.ger.name}, "
            f"FR -> {self.rule.fr.name}, "
            f"RUS -> {self.rule.rus.name}"
        )