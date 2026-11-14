from __future__ import annotations

import pytest

from routing.de_morgan_router import (
    DeMorganRouter,
    Destination,
    RoutingRule,
)


def _rule_one() -> RoutingRule:
    return RoutingRule(
        ger=Destination.STORAGE_A,
        fr=Destination.STORAGE_B,
        rus=Destination.RETURN_BUFFER,
    )


def _rule_two() -> RoutingRule:
    return RoutingRule(
        ger=Destination.RETURN_BUFFER,
        fr=Destination.STORAGE_A,
        rus=Destination.STORAGE_B,
    )


def test_destination_enum_values() -> None:
    assert Destination.STORAGE_A == 0
    assert Destination.STORAGE_B == 1
    assert Destination.RETURN_BUFFER == 2


def test_routing_rule_maps_each_language() -> None:
    rule = _rule_one()

    assert rule.destination("GER") is Destination.STORAGE_A
    assert rule.destination("FR") is Destination.STORAGE_B
    assert rule.destination("RUS") is Destination.RETURN_BUFFER


def test_routing_rule_as_dict() -> None:
    rule = _rule_one()

    assert rule.as_dict() == {
        "GER": Destination.STORAGE_A,
        "FR": Destination.STORAGE_B,
        "RUS": Destination.RETURN_BUFFER,
    }


def test_routing_rule_rejects_unknown_language() -> None:
    rule = _rule_one()

    with pytest.raises(ValueError, match="Unknown language"):
        rule.destination("ENG")


def test_router_routes_using_active_rule() -> None:
    rule = _rule_one()
    router = DeMorganRouter(rule=rule)

    assert router.route("GER") is Destination.STORAGE_A
    assert router.route("FR") is Destination.STORAGE_B
    assert router.route("RUS") is Destination.RETURN_BUFFER


def test_router_mapping_reflects_active_rule() -> None:
    router = DeMorganRouter(rule=_rule_one())

    assert router.mapping == {
        "GER": Destination.STORAGE_A,
        "FR": Destination.STORAGE_B,
        "RUS": Destination.RETURN_BUFFER,
    }


def test_router_set_rule_replaces_active_rule() -> None:
    router = DeMorganRouter(rule=_rule_one())

    router.set_rule(_rule_two())

    assert router.route("GER") is Destination.RETURN_BUFFER
    assert router.route("FR") is Destination.STORAGE_A
    assert router.route("RUS") is Destination.STORAGE_B


def test_router_mapping_changes_with_rule() -> None:
    router = DeMorganRouter(rule=_rule_one())

    first_mapping = router.mapping

    router.set_rule(_rule_two())

    second_mapping = router.mapping

    assert first_mapping != second_mapping
    assert second_mapping == {
        "GER": Destination.RETURN_BUFFER,
        "FR": Destination.STORAGE_A,
        "RUS": Destination.STORAGE_B,
    }


def test_router_describe() -> None:
    router = DeMorganRouter(rule=_rule_one())

    assert router.describe() == (
        "GER -> STORAGE_A, "
        "FR -> STORAGE_B, "
        "RUS -> RETURN_BUFFER"
    )


def test_router_and_routing_rule_are_deterministic() -> None:
    router = DeMorganRouter(rule=_rule_one())

    for language in ("GER", "FR", "RUS"):
        first = router.route(language)
        second = router.route(language)

        assert first is second