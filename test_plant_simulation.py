from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import PlantSimulationSettings
from simulation.plant_simulation import PlantSimulation


class FakeRemoteControl:
    """Minimal fake of the Plant Simulation RemoteControl interface."""

    def __init__(self) -> None:
        self.visible = False
        self.loaded_model: str | None = None
        self.closed_model = False
        self.started = False
        self.stopped = False
        self.reset = False
        self.stepped = False
        self.quit_called = False

        self.values: dict[str, object] = {}

    def SetVisible(self, visible: bool) -> None:
        self.visible = visible

    def LoadModel(self, path: str) -> None:
        self.loaded_model = path

    def CloseModel(self) -> None:
        self.closed_model = True

    def ResetSimulation(self, path: str) -> None:
        self.reset = True

    def StartSimulation(self, path: str) -> None:
        self.started = True

    def StopSimulation(self) -> None:
        self.stopped = True

    def StepSimulation(self, path: str) -> None:
        self.stepped = True

    def IsSimulationRunning(self) -> bool:
        return self.started and not self.stopped

    def HasSimulationError(self) -> bool:
        return False

    def GetValue(self, path: str) -> object:
        return self.values[path]

    def SetValue(self, path: str, value: object) -> None:
        self.values[path] = value

    def ExecuteSimTalk(self, code: str) -> object:
        return f"executed: {code}"

    def Quit(self) -> None:
        self.quit_called = True


@pytest.fixture
def fake_remote_control() -> FakeRemoteControl:
    return FakeRemoteControl()


@pytest.fixture
def plant_simulation(
    monkeypatch: pytest.MonkeyPatch,
    fake_remote_control: FakeRemoteControl,
    tmp_path: Path,
) -> PlantSimulation:
    monkeypatch.setattr(
        "win32com.client.Dispatch",
        lambda prog_id: fake_remote_control,
    )

    model = tmp_path / "test_model.spp"
    model.touch()

    settings = PlantSimulationSettings(
        model=model,
        visible=True,
        load_settle_time=0.0,
        diagnose_com=False,
    )

    return PlantSimulation(settings)


def test_connect(
    plant_simulation: PlantSimulation,
    fake_remote_control: FakeRemoteControl,
) -> None:
    plant_simulation.connect()

    assert plant_simulation.connected
    assert fake_remote_control.visible


def test_connect_is_idempotent(
    plant_simulation: PlantSimulation,
    fake_remote_control: FakeRemoteControl,
) -> None:
    plant_simulation.connect()
    first_connection = plant_simulation._remote_control

    plant_simulation.connect()

    assert plant_simulation._remote_control is first_connection


def test_load_model(
    plant_simulation: PlantSimulation,
    fake_remote_control: FakeRemoteControl,
) -> None:
    plant_simulation.connect()
    plant_simulation.load_model()

    assert fake_remote_control.loaded_model is not None
    assert fake_remote_control.loaded_model.endswith(
        "test_model.spp"
    )

    assert (
        fake_remote_control.values[
            ".Models.Model.RLConnection.on"
        ]
        is True
    )


def test_missing_model_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRemoteControl()

    monkeypatch.setattr(
        "win32com.client.Dispatch",
        lambda prog_id: fake,
    )

    settings = PlantSimulationSettings(
        model=Path("does_not_exist.spp"),
        load_settle_time=0.0,
        diagnose_com=False,
    )

    plant_simulation = PlantSimulation(settings)
    plant_simulation.connect()

    with pytest.raises(FileNotFoundError):
        plant_simulation.load_model()


def test_simulation_control(
    plant_simulation: PlantSimulation,
    fake_remote_control: FakeRemoteControl,
) -> None:
    plant_simulation.connect()

    plant_simulation.reset_simulation()
    plant_simulation.start_simulation()

    assert fake_remote_control.reset
    assert fake_remote_control.started
    assert plant_simulation.is_simulation_running()

    plant_simulation.step_simulation()
    assert fake_remote_control.stepped

    plant_simulation.stop_simulation()

    assert fake_remote_control.stopped
    assert not plant_simulation.is_simulation_running()


def test_value_access(
    plant_simulation: PlantSimulation,
    fake_remote_control: FakeRemoteControl,
) -> None:
    plant_simulation.connect()

    fake_remote_control.values[
        ".Models.Model.StorageA.numMU"
    ] = 8

    value = plant_simulation.get_value(
        ".Models.Model.StorageA.numMU"
    )

    assert value == 8

    plant_simulation.set_value(
        ".Models.Model.StorageA.numMU",
        9,
    )

    assert fake_remote_control.values[
        ".Models.Model.StorageA.numMU"
    ] == 9


def test_model_relative_value_access(
    plant_simulation: PlantSimulation,
    fake_remote_control: FakeRemoteControl,
) -> None:
    plant_simulation.connect()

    fake_remote_control.values[
        ".Models.Model.StorageA.numMU"
    ] = 8

    assert (
        plant_simulation.model_value(
            ".StorageA.numMU"
        )
        == 8
    )

    plant_simulation.set_model_value(
        ".StorageA.numMU",
        9,
    )

    assert fake_remote_control.values[
        ".Models.Model.StorageA.numMU"
    ] == 9


def test_execute_simtalk(
    plant_simulation: PlantSimulation,
) -> None:
    plant_simulation.connect()

    result = plant_simulation.execute_simtalk(
        "print 42"
    )

    assert result == "executed: print 42"


def test_disconnect(
    plant_simulation: PlantSimulation,
) -> None:
    plant_simulation.connect()
    assert plant_simulation.connected

    plant_simulation.disconnect()

    assert not plant_simulation.connected


def test_quit(
    plant_simulation: PlantSimulation,
    fake_remote_control: FakeRemoteControl,
) -> None:
    plant_simulation.connect()

    plant_simulation.quit()

    assert fake_remote_control.quit_called
    assert not plant_simulation.connected


def test_operations_require_connection(
    plant_simulation: PlantSimulation,
) -> None:
    with pytest.raises(RuntimeError):
        plant_simulation.start_simulation()

    with pytest.raises(RuntimeError):
        plant_simulation.get_value(
            ".Models.Model.StorageA.numMU"
        )


def test_enable_rl_connection(
    plant_simulation: PlantSimulation,
    fake_remote_control: FakeRemoteControl,
) -> None:
    plant_simulation.connect()

    plant_simulation.enable_rl_connection()

    assert (
        fake_remote_control.values[
            ".Models.Model.RLConnection.on"
        ]
        is True
    )


def test_disable_rl_connection(
    plant_simulation: PlantSimulation,
    fake_remote_control: FakeRemoteControl,
) -> None:
    plant_simulation.connect()

    plant_simulation.disable_rl_connection()

    assert (
        fake_remote_control.values[
            ".Models.Model.RLConnection.on"
        ]
        is False
    )

def test_read_event_controller_signals(monkeypatch) -> None:
    plant = PlantSimulation(
        PlantSimulationSettings()
    )

    values = {
        ".Models.Model.EventController.InReset": False,
        ".Models.Model.EventController.IsRunning": True,
        ".Models.Model.EventController.SimTime": 42.5,
    }

    def fake_get_value(path: str):
        return values[path]

    monkeypatch.setattr(
        plant,
        "get_value",
        fake_get_value,
    )

    signals = plant.read_event_controller_signals()

    assert signals == (
        False,
        True,
        42.5,
    )