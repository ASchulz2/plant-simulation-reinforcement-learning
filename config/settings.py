
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os


# Load local configuration from the project-root .env file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class PlantSimulationSettings:
    """
    Configuration for the Plant Simulation COM interface.

    Plant Simulation model paths are provided through environment
    variables, keeping machine-specific configuration outside the
    repository.
    """

    model: Path | None = None

    prog_id: str = (
        "Tecnomatix.PlantSimulation.RemoteControl.16.1"
    )

    model_path: str = ".Models.Model"

    visible: bool = True
    load_settle_time: float = 2.0

    com_warning_threshold: float = 0.01
    diagnose_com: bool = True

    def __post_init__(self) -> None:
        configured_model = (
            self.model
            if self.model is not None
            else os.getenv("PLANT_SIMULATION_MODEL")
        )

        if configured_model is not None:
            object.__setattr__(
                self,
                "model",
                Path(configured_model).expanduser(),
            )
