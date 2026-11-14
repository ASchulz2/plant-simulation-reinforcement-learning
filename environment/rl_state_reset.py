from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLStateResetSettings:
    """
    Configuration for resetting the Plant Simulation RL state.
    """

    robot_path: str = (
        ".Models.Model.PickAndPlaceRobot"
    )

    initial_decision_point_id: int = 0
    initial_action: int = -1
    initial_action_received: bool = False
    initial_decision_point: bool = False
    initial_episode_finished: bool = False
    initial_episode_success_pending: bool = False


class RLStateResetter:
    """
    Reset the RL protocol state of PickAndPlaceRobot.

    This class deliberately knows nothing about:
        - TCP communication
        - episode control
        - rewards
        - observations
        - simulation start/stop

    It only restores the Plant Simulation attributes required
    for starting a fresh RL episode.
    """

    def __init__(
        self,
        plant,
        settings: RLStateResetSettings | None = None,
    ) -> None:
        self.plant = plant
        self.settings = (
            settings
            if settings is not None
            else RLStateResetSettings()
        )

    def reset(self) -> None:
        """
        Restore PickAndPlaceRobot to its initial RL protocol state.
        """

        robot_path = self.settings.robot_path

        self.plant.set_value(
            f"{robot_path}.decisionPointID",
            self.settings.initial_decision_point_id,
        )

        self.plant.set_value(
            f"{robot_path}.action",
            self.settings.initial_action,
        )

        self.plant.set_value(
            f"{robot_path}.actionReceived",
            self.settings.initial_action_received,
        )

        self.plant.set_value(
            f"{robot_path}.decisionPoint",
            self.settings.initial_decision_point,
        )

        self.plant.set_value(
            f"{robot_path}.episodeFinished",
            self.settings.initial_episode_finished,
        )

        self.plant.set_value(
            f"{robot_path}.episodeSuccessPending",
            self.settings.initial_episode_success_pending,
        )