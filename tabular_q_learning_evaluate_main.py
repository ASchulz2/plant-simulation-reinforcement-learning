from __future__ import annotations

from pathlib import Path

from config.settings import PlantSimulationSettings
from config.tcp_settings import TCPSettings
from tabular_q_learning_evaluate import (
    evaluate,
)


def main() -> None:
    """Start tabular Q-learning checkpoint evaluation."""

    plant_settings = PlantSimulationSettings(
        diagnose_com=False,
        visible=True,
    )

    tcp_settings = TCPSettings(
        host="127.0.0.1",
        port=5000,
        connect_timeout=5.0,
        receive_timeout=60.0,
        backlog=1,
    )

    checkpoint_path = (
        Path(__file__).resolve().parent
        / "checkpoint"
        / "tabular_q_learning_pure.pkl"
    )

    evaluate(
        plant_settings=plant_settings,
        tcp_settings=tcp_settings,
        num_episodes=1,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
