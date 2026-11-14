from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import win32com.client

from config.settings import PlantSimulationSettings

EVENT_CONTROLLER_PATH = ".Models.Model.EventController"

@dataclass
class PlantSimulation:
    """
    Python interface to Plant Simulation through RemoteControl COM.

    Responsibilities
    ----------------
    - connection management
    - model lifecycle
    - simulation control
    - value access
    - SimTalk execution
    - COM diagnostics

    Synchronization with the Python application is deliberately handled
    outside this class by the TCP layer.
    """

    settings: PlantSimulationSettings

    _remote_control: Any | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """Return whether a COM connection exists."""
        return self._remote_control is not None

    def connect(self) -> None:
        """Connect to Plant Simulation through RemoteControl."""

        if self.connected:
            return

        self._remote_control = win32com.client.Dispatch(
            self.settings.prog_id
        )

        if self.settings.visible:
            self._remote_control.SetVisible(True)

    def disconnect(self) -> None:
        """Release the COM reference."""
        self._remote_control = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def enable_rl_connection(self) -> None:
        """
        Enable the RLConnection socket in the loaded Plant Simulation model.

        The RLConnection object is the server-side TCP endpoint used by
        the Python RL environment.
        """

        self._require_connection()

        self.set_value(
            f"{self.settings.model_path}.RLConnection.on",
            True,
        )

    def disable_rl_connection(self) -> None:
        """
        Disable the RLConnection socket.
        """

        self._require_connection()

        self.set_value(
            f"{self.settings.model_path}.RLConnection.on",
            False,
        )

    def load_model(self) -> None:
        """Load the configured Plant Simulation model."""

        self._require_connection()

        if self.settings.model is None:
            raise RuntimeError(
                "Plant Simulation model is not configured. "
                "Set PLANT_SIMULATION_MODEL in your .env file."
            )

        model = self.settings.model.resolve()

        if not model.is_file():
            raise FileNotFoundError(
                f"Plant Simulation model not found: {model}"
            )

        self._remote_control.LoadModel(str(model))

        time.sleep(self.settings.load_settle_time)

        self.enable_rl_connection()

    def close_model(self) -> None:
        """Close the currently loaded model."""

        self._require_connection()

        self._remote_control.CloseModel()

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def reset_simulation(self) -> None:
        """Reset the configured Plant Simulation model."""

        self._require_connection()

        self._remote_control.ResetSimulation(
            self.settings.model_path
        )

    def start_simulation(self) -> None:
        """Start the configured Plant Simulation model."""

        self._require_connection()

        self._remote_control.StartSimulation(
            self.settings.model_path
        )

    def stop_simulation(self) -> None:
        """Stop the simulation."""

        self._require_connection()

        self._remote_control.StopSimulation()

    def step_simulation(self) -> None:
        """Advance the simulation by one simulation step."""

        self._require_connection()

        self._remote_control.StepSimulation(
            self.settings.model_path
        )

    def is_simulation_running(self) -> bool:
        """Return whether Plant Simulation is currently running."""

        self._require_connection()

        return bool(
            self._remote_control.IsSimulationRunning()
        )

    def has_simulation_error(self) -> bool:
        """Return whether Plant Simulation reports an error."""

        self._require_connection()

        return bool(
            self._remote_control.HasSimulationError()
        )

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------

    def get_value(self, path: str) -> Any:
        """
        Get a value from an absolute Plant Simulation path.

        Example
        -------
        .Models.Model.StorageA.numMU
        """

        self._require_connection()

        start = time.perf_counter()

        try:
            return self._remote_control.GetValue(path)

        finally:
            elapsed = time.perf_counter() - start

            if (
                self.settings.diagnose_com
                and elapsed >= self.settings.com_warning_threshold
            ):
                print(
                    "[COM] Slow GetValue: "
                    f"{elapsed * 1000.0:.3f} ms | "
                    f"{path}"
                )

    def set_value(
        self,
        path: str,
        value: Any,
    ) -> None:
        """Set a value at an absolute Plant Simulation path."""

        self._require_connection()

        start = time.perf_counter()

        try:
            self._remote_control.SetValue(
                path,
                value,
            )

        finally:
            elapsed = time.perf_counter() - start

            if (
                self.settings.diagnose_com
                and elapsed >= self.settings.com_warning_threshold
            ):
                print(
                    "[COM] Slow SetValue: "
                    f"{elapsed * 1000.0:.3f} ms | "
                    f"{path}"
                )

    # ------------------------------------------------------------------
    # Model-relative value access
    # ------------------------------------------------------------------

    def model_value(self, relative_path: str) -> Any:
        """
        Get a value relative to the model root.

        Example
        -------
        model_value(".StorageA.numMU")
        """

        return self.get_value(
            self._model_path(relative_path)
        )

    def set_model_value(
        self,
        relative_path: str,
        value: Any,
    ) -> None:
        """Set a value relative to the model root."""

        self.set_value(
            self._model_path(relative_path),
            value,
        )

    def _model_path(self, relative_path: str) -> str:
        """Build an absolute path from the model root."""

        if not relative_path.startswith("."):
            relative_path = "." + relative_path

        return (
            self.settings.model_path
            + relative_path
        )

    # ------------------------------------------------------------------
    # SimTalk
    # ------------------------------------------------------------------

    def execute_simtalk(self, code: str) -> Any:
        """Execute SimTalk through RemoteControl."""

        self._require_connection()

        return self._remote_control.ExecuteSimTalk(code)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_methods(self) -> None:
        """Print publicly exposed RemoteControl methods."""

        self._require_connection()

        for name in dir(self._remote_control):
            if not name.startswith("_"):
                print(name)

    def print_simulation_status(self) -> None:
        """Print basic Plant Simulation execution status."""

        running = self.is_simulation_running()
        error = self.has_simulation_error()

        print("[COM] Simulation running:", running)
        print("[COM] Simulation error:", error)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def quit(self) -> None:
        """Quit the Plant Simulation instance."""

        if not self.connected:
            return

        self._remote_control.Quit()
        self._remote_control = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_connection(self) -> None:
        """Raise if no COM connection exists."""

        if not self.connected:
            raise RuntimeError(
                "Plant Simulation is not connected."
            )

    def read_event_controller_signals(
        self,
    ) -> tuple[bool, bool, float]:
        """Read the EventController values required by the RL observer.

        Returns
        -------
        tuple[bool, bool, float]
            ``(in_reset, is_running, sim_time)``.

        The method only performs COM access. Interpretation of these
        signals belongs to ``EventControllerObserver``.
        """

        in_reset = bool(
            self.get_value(
                f"{EVENT_CONTROLLER_PATH}.InReset"
            )
        )

        is_running = bool(
            self.get_value(
                f"{EVENT_CONTROLLER_PATH}.IsRunning"
            )
        )

        sim_time = float(
            self.get_value(
                f"{EVENT_CONTROLLER_PATH}.SimTime"
            )
        )

        return (
            in_reset,
            is_running,
            sim_time,
        )