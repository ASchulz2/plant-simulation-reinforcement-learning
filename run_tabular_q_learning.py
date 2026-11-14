from __future__ import annotations

from config.settings import PlantSimulationSettings
from config.tcp_settings import TCPSettings
from tabular_q_learning_main import run


def main() -> None:
    """Start the real tabular Q-learning experiment."""

    plant_settings = PlantSimulationSettings(
        diagnose_com=False,
        visible=True
    )

    tcp_settings = TCPSettings(
        host="127.0.0.1",
        port=5000,
        connect_timeout=5.0,
        receive_timeout=60.0,
        backlog=1,
    )

    run(
        plant_settings=plant_settings,
        tcp_settings=tcp_settings,
        num_episodes=10,
        resume=False
    )


if __name__ == "__main__":
    main()