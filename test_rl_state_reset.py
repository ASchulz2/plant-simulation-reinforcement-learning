from __future__ import annotations

from dataclasses import dataclass, field

from environment.rl_state_reset import (
    RLStateResetSettings,
    RLStateResetter,
)


@dataclass
class FakePlant:
    values: dict[str, object] = field(
        default_factory=dict
    )

    def set_value(
        self,
        path: str,
        value: object,
    ) -> None:
        self.values[path] = value


def test_default_settings_use_correct_robot_path() -> None:
    settings = RLStateResetSettings()

    assert (
        settings.robot_path
        == ".Models.Model.PickAndPlaceRobot"
    )


def test_reset_writes_initial_robot_state() -> None:
    plant = FakePlant()

    resetter = RLStateResetter(
        plant=plant,
        settings=RLStateResetSettings(),
    )

    resetter.reset()

    expected = {
        ".Models.Model.PickAndPlaceRobot.decisionPointID": 0,
        ".Models.Model.PickAndPlaceRobot.action": -1,
        ".Models.Model.PickAndPlaceRobot.actionReceived": False,
        ".Models.Model.PickAndPlaceRobot.decisionPoint": False,
        ".Models.Model.PickAndPlaceRobot.episodeFinished": False,
        ".Models.Model.PickAndPlaceRobot.episodeSuccessPending": False,
    }

    assert plant.values == expected


def test_reset_respects_configured_initial_values() -> None:
    plant = FakePlant()

    settings = RLStateResetSettings(
        initial_decision_point_id=17,
        initial_action=2,
        initial_action_received=True,
        initial_decision_point=True,
        initial_episode_finished=True,
        initial_episode_success_pending=True,
    )

    resetter = RLStateResetter(
        plant=plant,
        settings=settings,
    )

    resetter.reset()

    expected = {
        ".Models.Model.PickAndPlaceRobot.decisionPointID": 17,
        ".Models.Model.PickAndPlaceRobot.action": 2,
        ".Models.Model.PickAndPlaceRobot.actionReceived": True,
        ".Models.Model.PickAndPlaceRobot.decisionPoint": True,
        ".Models.Model.PickAndPlaceRobot.episodeFinished": True,
        ".Models.Model.PickAndPlaceRobot.episodeSuccessPending": True,
    }

    assert plant.values == expected