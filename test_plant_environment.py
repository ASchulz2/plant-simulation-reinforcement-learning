from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from environment.emptying_event import EmptyingEvent
from environment.event_controller_observer import (
    EventControllerObservation,
    EventControllerObserver,
    EventControllerState,
)
from environment.plant_environment import (
    PlantEnvironment,
    PlantEnvironmentSettings,
)
from environment.return_event import ReturnEvent
from environment.termination import TerminationReason
from problem.observation import Observation
from problem.problem import ManufacturingProblem
from problem.tabular_state_encoder import TabularStateEncoder
from tcp.client import TCPClient
from tcp.connection import (
    ConnectionError as TCPConnectionError,
)
from tcp.handshake import HandshakeState
from tcp.protocol import (
    ObservationMessage,
    TerminalMessage,
)


def _observation(
    *,
    simulation_time: float = 1.0,
    throughput: int = 0,
    admission_buffer_contents: tuple[str, ...] = ("GER",),
    storage_a_count: int = 0,
    storage_a_language: str = "",
    storage_b_count: int = 0,
    storage_b_language: str = "",
) -> Observation:
    """Create a valid semantic observation for environment tests."""

    return Observation(
        simulation_time=simulation_time,
        admission_buffer_contents=admission_buffer_contents,
        storage_a_count=storage_a_count,
        storage_a_language=storage_a_language,
        storage_b_count=storage_b_count,
        storage_b_language=storage_b_language,
        throughput=throughput,
    )


def _event_controller_observation(
    *,
    state: EventControllerState,
    in_reset: bool,
    is_running: bool,
    sim_time: float,
) -> EventControllerObservation:
    """Create an EventController observation for lifecycle tests."""

    return EventControllerObservation(
        state=state,
        in_reset=in_reset,
        is_running=is_running,
        sim_time=sim_time,
    )


def _make_environment(
    *,
    max_steps: int = 1000,
) -> tuple[
    PlantEnvironment,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    """Create a PlantEnvironment with mocked dependencies."""

    plant = MagicMock()
    tcp_client = MagicMock()
    observation_source = MagicMock()
    observation_encoder = MagicMock()
    reward_calculator = MagicMock()
    emptying_event_reader = MagicMock()
    return_event_reader = MagicMock()

    # Use a real observer because its state-transition semantics are
    # part of the PlantEnvironment lifecycle contract.
    event_controller_observer = EventControllerObserver()

    problem = MagicMock(
        spec=ManufacturingProblem,
    )

    plant.read_event_controller_signals.return_value = (
        False,
        False,
        0.0,
    )

    environment = PlantEnvironment(
        plant=plant,
        tcp_client=tcp_client,
        observation_source=observation_source,
        observation_encoder=observation_encoder,
        reward_calculator=reward_calculator,
        problem=problem,
        event_controller_observer=event_controller_observer,
        emptying_event_reader=emptying_event_reader,
        return_event_reader=return_event_reader,
        settings=PlantEnvironmentSettings(
            max_steps=max_steps,
        ),
    )

    # Existing tests use a mocked problem. Keep feasibility neutral for
    # those tests and let the dedicated feasibility tests install the
    # real ManufacturingProblem + TabularStateEncoder.
    environment.tabular_state_encoder = MagicMock()

    problem.feasible_actions.return_value = (
        0,
        1,
        2,
    )

    return (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    )


def _prepare_running_environment(
    environment: PlantEnvironment,
    *,
    semantic_observation: Observation,
    encoded_observation: torch.Tensor,
    decision_point_id: int = 1,
) -> None:
    """Place the environment in a valid pre-step state."""

    environment._running = True

    environment._current_decision_point_id = (
        decision_point_id
    )

    environment._current_semantic_observation = (
        semantic_observation
    )

    environment._current_observation = (
        encoded_observation
    )


def _neutral_tensors() -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Provide neutral PPO-compatible transition fields."""

    return (
        torch.tensor(0.0),
        torch.tensor(0.0),
    )


def _configure_ack(
    tcp_client: MagicMock,
    *,
    decision_point_id: int,
) -> None:
    acknowledgement = MagicMock()
    acknowledgement.decision_point_id = (
        decision_point_id
    )

    tcp_client.receive_acknowledgement.return_value = (
        acknowledgement
    )


# ======================================================================
# Settings
# ======================================================================


def test_settings_defaults() -> None:
    settings = PlantEnvironmentSettings()

    assert settings.robot_path == (
        ".Models.Model.PickAndPlaceRobot"
    )
    assert settings.max_steps == 1000
    assert settings.poll_interval == 0.001
    assert settings.lifecycle_timeout == 60.0


def test_settings_reject_invalid_max_steps() -> None:
    with pytest.raises(
        TypeError,
        match="max_steps must be an integer",
    ):
        PlantEnvironmentSettings(
            max_steps=1.5,
        )

    with pytest.raises(
        ValueError,
        match="max_steps must be positive",
    ):
        PlantEnvironmentSettings(
            max_steps=0,
        )


def test_settings_reject_invalid_poll_interval() -> None:
    with pytest.raises(
        TypeError,
        match="poll_interval must be numeric",
    ):
        PlantEnvironmentSettings(
            poll_interval="0.1",
        )

    with pytest.raises(
        ValueError,
        match="poll_interval must be positive",
    ):
        PlantEnvironmentSettings(
            poll_interval=0.0,
        )


def test_settings_reject_invalid_lifecycle_timeout() -> None:
    with pytest.raises(
        TypeError,
        match="lifecycle_timeout must be numeric",
    ):
        PlantEnvironmentSettings(
            lifecycle_timeout="1.0",
        )

    with pytest.raises(
        ValueError,
        match="lifecycle_timeout must be positive",
    ):
        PlantEnvironmentSettings(
            lifecycle_timeout=0.0,
        )


# ======================================================================
# Current state
# ======================================================================


def test_current_state_returns_semantic_observation() -> None:
    (
        environment,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _make_environment()

    semantic = _observation()
    encoded = torch.tensor([0.25])

    _prepare_running_environment(
        environment,
        semantic_observation=semantic,
        encoded_observation=encoded,
    )

    assert environment.current_state is semantic
    assert environment.current_observation is encoded


# ======================================================================
# Normal step + events
# ======================================================================


def test_normal_step_passes_events_to_reward() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    previous = _observation(
        simulation_time=1.0,
        throughput=10,
    )

    next_state = _observation(
        simulation_time=2.0,
        throughput=11,
    )

    previous_encoded = torch.tensor([1.0])
    next_encoded = torch.tensor([2.0])

    emptying_event = EmptyingEvent(
        storage="B",
        event_id=1,
        mode="Drain",
        count=3,
        total_waiting_time=105.0,
    )

    return_event = ReturnEvent(
        event_id=1,
        language="FR",
        waiting_time=41.6623,
    )

    emptying_event_reader.read.return_value = (
        emptying_event,
    )

    return_event_reader.read.return_value = (
        return_event,
    )

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=1,
    )

    environment.problem.validate_action.return_value = None

    _configure_ack(
        tcp_client,
        decision_point_id=1,
    )

    tcp_client.receive_episode_message.return_value = (
        ObservationMessage(
            decision_point_id=2,
            terminal=False,
        )
    )

    observation_source.read.return_value = next_state
    observation_encoder.encode.return_value = next_encoded
    reward_calculator.calculate.return_value = 3.0

    log_probability, value = _neutral_tensors()

    transition = environment.step(
        0,
        log_probability,
        value,
    )

    assert transition.terminated is False
    assert transition.reward == 3.0

    emptying_event_reader.read.assert_called_once()
    return_event_reader.read.assert_called_once()

    reward_calculator.calculate.assert_called_once_with(
        previous,
        next_state,
        action=0,
        action_feasible=True,
        emptying_events=(emptying_event,),
        return_events=(return_event,),
        terminated=False,
        succeeded=False,
    )

    assert environment.running is True
    assert environment.step_count == 1

    plant.stop_simulation.assert_not_called()


# ======================================================================
# Action feasibility
# ======================================================================


def test_normal_step_passes_true_when_action_is_feasible() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    problem = ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        admission_buffer_capacity=10,
        actions=(0, 1, 2),
    )

    environment.problem = problem
    environment.tabular_state_encoder = (
        TabularStateEncoder(
            problem=problem,
        )
    )

    previous = _observation(
        simulation_time=1.0,
        throughput=10,
        admission_buffer_contents=("GER",),
        storage_a_count=2,
        storage_a_language="GER",
        storage_b_count=0,
        storage_b_language="",
    )

    next_state = _observation(
        simulation_time=2.0,
        throughput=11,
        admission_buffer_contents=("FR",),
        storage_a_count=3,
        storage_a_language="GER",
        storage_b_count=0,
        storage_b_language="",
    )

    previous_encoded = torch.tensor([1.0])
    next_encoded = torch.tensor([2.0])

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=1,
    )

    _configure_ack(
        tcp_client,
        decision_point_id=1,
    )

    tcp_client.receive_episode_message.return_value = (
        ObservationMessage(
            decision_point_id=2,
            terminal=False,
        )
    )

    observation_source.read.return_value = next_state
    observation_encoder.encode.return_value = next_encoded

    emptying_event_reader.read.return_value = ()
    return_event_reader.read.return_value = ()

    reward_calculator.calculate.return_value = 1.0

    log_probability, value = _neutral_tensors()

    transition = environment.step(
        0,
        log_probability,
        value,
    )

    assert transition.reward == 1.0

    reward_calculator.calculate.assert_called_once_with(
        previous,
        next_state,
        action=0,
        action_feasible=True,
        emptying_events=(),
        return_events=(),
        terminated=False,
        succeeded=False,
    )


def test_normal_step_passes_false_when_action_is_infeasible() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    problem = ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        admission_buffer_capacity=10,
        actions=(0, 1, 2),
    )

    environment.problem = problem
    environment.tabular_state_encoder = (
        TabularStateEncoder(
            problem=problem,
        )
    )

    previous = _observation(
        simulation_time=1.0,
        throughput=10,
        admission_buffer_contents=("GER",),
        storage_a_count=2,
        storage_a_language="FR",
        storage_b_count=0,
        storage_b_language="",
    )

    next_state = _observation(
        simulation_time=2.0,
        throughput=10,
        admission_buffer_contents=("FR",),
        storage_a_count=2,
        storage_a_language="FR",
        storage_b_count=1,
        storage_b_language="GER",
    )

    previous_encoded = torch.tensor([1.0])
    next_encoded = torch.tensor([2.0])

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=1,
    )

    _configure_ack(
        tcp_client,
        decision_point_id=1,
    )

    tcp_client.receive_episode_message.return_value = (
        ObservationMessage(
            decision_point_id=2,
            terminal=False,
        )
    )

    observation_source.read.return_value = next_state
    observation_encoder.encode.return_value = next_encoded

    emptying_event_reader.read.return_value = ()
    return_event_reader.read.return_value = ()

    reward_calculator.calculate.return_value = -1.0

    log_probability, value = _neutral_tensors()

    transition = environment.step(
        0,
        log_probability,
        value,
    )

    assert transition.reward == -1.0

    reward_calculator.calculate.assert_called_once_with(
        previous,
        next_state,
        action=0,
        action_feasible=False,
        emptying_events=(),
        return_events=(),
        terminated=False,
        succeeded=False,
    )


def test_normal_step_uses_previous_state_for_feasibility() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    problem = ManufacturingProblem(
        target_throughput=1000,
        storage_capacity=10,
        admission_buffer_capacity=10,
        actions=(0, 1, 2),
    )

    environment.problem = problem
    environment.tabular_state_encoder = (
        TabularStateEncoder(
            problem=problem,
        )
    )

    # Action 0 is infeasible in the previous state:
    #
    #     current language = GER
    #     Storage A       = FR
    #
    # But the next state makes action 0 feasible:
    #
    #     current language = FR
    #     Storage A       = FR
    #
    # The environment must nevertheless classify the selected action
    # according to the state in which the decision was made.
    previous = _observation(
        simulation_time=1.0,
        throughput=10,
        admission_buffer_contents=("GER",),
        storage_a_count=2,
        storage_a_language="FR",
        storage_b_count=0,
        storage_b_language="",
    )

    next_state = _observation(
        simulation_time=2.0,
        throughput=10,
        admission_buffer_contents=("FR",),
        storage_a_count=2,
        storage_a_language="FR",
        storage_b_count=0,
        storage_b_language="",
    )

    previous_encoded = torch.tensor([1.0])
    next_encoded = torch.tensor([2.0])

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=1,
    )

    _configure_ack(
        tcp_client,
        decision_point_id=1,
    )

    tcp_client.receive_episode_message.return_value = (
        ObservationMessage(
            decision_point_id=2,
            terminal=False,
        )
    )

    observation_source.read.return_value = next_state
    observation_encoder.encode.return_value = next_encoded

    emptying_event_reader.read.return_value = ()
    return_event_reader.read.return_value = ()

    reward_calculator.calculate.return_value = -1.0

    log_probability, value = _neutral_tensors()

    environment.step(
        0,
        log_probability,
        value,
    )

    reward_calculator.calculate.assert_called_once_with(
        previous,
        next_state,
        action=0,
        action_feasible=False,
        emptying_events=(),
        return_events=(),
        terminated=False,
        succeeded=False,
    )


# ======================================================================
# Horizon termination
# ======================================================================


def test_horizon_terminal_sets_reason() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment(
        max_steps=1,
    )

    previous = _observation(
        simulation_time=1.0,
        throughput=10,
    )

    next_state = _observation(
        simulation_time=2.0,
        throughput=11,
    )

    previous_encoded = torch.tensor([1.0])
    next_encoded = torch.tensor([2.0])

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=1,
    )

    environment.problem.validate_action.return_value = None

    _configure_ack(
        tcp_client,
        decision_point_id=1,
    )

    tcp_client.receive_episode_message.return_value = (
        ObservationMessage(
            decision_point_id=2,
            terminal=False,
        )
    )

    observation_source.read.return_value = next_state
    observation_encoder.encode.return_value = next_encoded

    reward_calculator.calculate.return_value = 3.0

    emptying_event_reader.read.return_value = ()
    return_event_reader.read.return_value = ()

    log_probability, value = _neutral_tensors()

    transition = environment.step(
        0,
        log_probability,
        value,
    )

    assert transition.terminated is True
    assert transition.succeeded is False
    assert transition.progress_failure is False

    assert (
        transition.termination_reason
        is TerminationReason.HORIZON_TIMEOUT
    )

    assert (
        environment.termination_reason
        is TerminationReason.HORIZON_TIMEOUT
    )

    assert environment.running is False
    assert environment.step_count == 1

    plant.stop_simulation.assert_called_once()

    reward_calculator.calculate.assert_called_once_with(
        previous,
        next_state,
        action=0,
        action_feasible=True,
        emptying_events=(),
        return_events=(),
        terminated=True,
        succeeded=False,
    )


# ======================================================================
# Manufacturing success
# ======================================================================


def test_success_terminal_sets_reason_and_passes_events() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    previous = _observation(
        simulation_time=1.0,
        throughput=10,
    )

    terminal_state = _observation(
        simulation_time=3.0,
        throughput=1000,
    )

    previous_encoded = torch.tensor([1.0])
    terminal_encoded = torch.tensor([3.0])

    emptying_event = EmptyingEvent(
        storage="A",
        event_id=1,
        mode="Drain",
        count=3,
        total_waiting_time=172.0,
    )

    return_event = ReturnEvent(
        event_id=4,
        language="FR",
        waiting_time=39.0172,
    )

    emptying_event_reader.read.return_value = (
        emptying_event,
    )

    return_event_reader.read.return_value = (
        return_event,
    )

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=5,
    )

    environment.problem.validate_action.return_value = None

    _configure_ack(
        tcp_client,
        decision_point_id=5,
    )

    tcp_client.receive_episode_message.return_value = (
        TerminalMessage(
            decision_point_id=5,
            reason="success",
        )
    )

    observation_source.read.return_value = terminal_state
    observation_encoder.encode.return_value = terminal_encoded
    reward_calculator.calculate.return_value = 12.0

    log_probability, value = _neutral_tensors()

    transition = environment.step(
        1,
        log_probability,
        value,
    )

    assert transition.terminated is True
    assert transition.succeeded is True
    assert transition.progress_failure is False

    assert (
        transition.termination_reason
        is TerminationReason.SUCCESS
    )

    assert (
        environment.termination_reason
        is TerminationReason.SUCCESS
    )

    assert environment.running is False

    plant.stop_simulation.assert_called_once()

    emptying_event_reader.read.assert_called_once()
    return_event_reader.read.assert_called_once()

    reward_calculator.calculate.assert_called_once_with(
        previous,
        terminal_state,
        action=1,
        action_feasible=True,
        emptying_events=(emptying_event,),
        return_events=(return_event,),
        terminated=True,
        succeeded=True,
    )


# ======================================================================
# Manufacturing failure
# ======================================================================


def test_manufacturing_failure_terminal_sets_reason() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    previous = _observation(
        simulation_time=1.0,
        throughput=10,
    )

    terminal_state = _observation(
        simulation_time=2.5,
        throughput=20,
    )

    previous_encoded = torch.tensor([1.0])
    terminal_encoded = torch.tensor([2.5])

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=7,
    )

    environment.problem.validate_action.return_value = None

    _configure_ack(
        tcp_client,
        decision_point_id=7,
    )

    tcp_client.receive_episode_message.return_value = (
        TerminalMessage(
            decision_point_id=7,
            reason="failure",
        )
    )

    observation_source.read.return_value = terminal_state
    observation_encoder.encode.return_value = terminal_encoded
    reward_calculator.calculate.return_value = -10.0

    emptying_event_reader.read.return_value = ()
    return_event_reader.read.return_value = ()

    log_probability, value = _neutral_tensors()

    transition = environment.step(
        2,
        log_probability,
        value,
    )

    assert transition.terminated is True
    assert transition.succeeded is False
    assert transition.progress_failure is False

    assert (
        transition.termination_reason
        is TerminationReason.MANUFACTURING_FAILURE
    )

    assert (
        environment.termination_reason
        is TerminationReason.MANUFACTURING_FAILURE
    )

    assert environment.running is False

    plant.stop_simulation.assert_called_once()

    reward_calculator.calculate.assert_called_once_with(
        previous,
        terminal_state,
        action=2,
        action_feasible=True,
        emptying_events=(),
        return_events=(),
        terminated=True,
        succeeded=False,
    )


# ======================================================================
# Communication failure
# ======================================================================


def test_communication_failure_reuses_previous_observation() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    previous = _observation(
        simulation_time=4.0,
        throughput=12,
    )

    previous_encoded = torch.tensor([4.0])

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=3,
    )

    environment.problem.validate_action.return_value = None

    tcp_client.send_action.side_effect = (
        TCPConnectionError(
            "test transport failure"
        )
    )

    reward_calculator.calculate.return_value = -10.0

    log_probability, value = _neutral_tensors()

    transition = environment.step(
        0,
        log_probability,
        value,
    )

    assert transition.terminated is True
    assert transition.succeeded is False
    assert transition.progress_failure is True

    assert (
        transition.termination_reason
        is TerminationReason.COMMUNICATION_FAILURE
    )

    assert transition.state is transition.next_state
    assert transition.elapsed_time == 0.0

    assert (
        transition.next_throughput
        == previous.throughput
    )

    assert (
        environment.termination_reason
        is TerminationReason.COMMUNICATION_FAILURE
    )

    assert environment.running is False

    plant.stop_simulation.assert_called_once()

    observation_source.read.assert_not_called()
    observation_encoder.encode.assert_not_called()
    emptying_event_reader.read.assert_not_called()
    return_event_reader.read.assert_not_called()

    reward_calculator.calculate.assert_called_once_with(
        previous,
        previous,
        action=0,
        action_feasible=True,
        emptying_events=(),
        return_events=(),
        terminated=True,
        succeeded=False,
    )


def test_terminal_reason_mismatch_becomes_communication_failure() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    previous = _observation(
        simulation_time=4.0,
        throughput=12,
    )

    previous_encoded = torch.tensor([4.0])

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=3,
    )

    environment.problem.validate_action.return_value = None

    _configure_ack(
        tcp_client,
        decision_point_id=3,
    )

    tcp_client.receive_episode_message.return_value = (
        TerminalMessage(
            decision_point_id=999,
            reason="success",
        )
    )

    reward_calculator.calculate.return_value = -10.0

    log_probability, value = _neutral_tensors()

    transition = environment.step(
        0,
        log_probability,
        value,
    )

    assert (
        transition.termination_reason
        is TerminationReason.COMMUNICATION_FAILURE
    )

    assert transition.progress_failure is True
    assert transition.succeeded is False
    assert transition.elapsed_time == 0.0

    plant.stop_simulation.assert_called_once()

    observation_source.read.assert_not_called()
    observation_encoder.encode.assert_not_called()
    emptying_event_reader.read.assert_not_called()
    return_event_reader.read.assert_not_called()

    reward_calculator.calculate.assert_called_once_with(
        previous,
        previous,
        action=0,
        action_feasible=True,
        emptying_events=(),
        return_events=(),
        terminated=True,
        succeeded=False,
    )


def test_non_increasing_decision_point_id_is_communication_failure() -> None:
    (
        environment,
        plant,
        tcp_client,
        observation_source,
        observation_encoder,
        reward_calculator,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    previous = _observation(
        simulation_time=4.0,
        throughput=12,
    )

    previous_encoded = torch.tensor([4.0])

    _prepare_running_environment(
        environment,
        semantic_observation=previous,
        encoded_observation=previous_encoded,
        decision_point_id=3,
    )

    environment.problem.validate_action.return_value = None

    _configure_ack(
        tcp_client,
        decision_point_id=3,
    )

    tcp_client.receive_episode_message.return_value = (
        ObservationMessage(
            decision_point_id=3,
            terminal=False,
        )
    )

    reward_calculator.calculate.return_value = -10.0

    log_probability, value = _neutral_tensors()

    transition = environment.step(
        1,
        log_probability,
        value,
    )

    assert (
        transition.termination_reason
        is TerminationReason.COMMUNICATION_FAILURE
    )

    assert transition.progress_failure is True
    assert transition.succeeded is False
    assert transition.elapsed_time == 0.0

    observation_source.read.assert_not_called()
    observation_encoder.encode.assert_not_called()
    emptying_event_reader.read.assert_not_called()
    return_event_reader.read.assert_not_called()

    plant.stop_simulation.assert_called_once()

    reward_calculator.calculate.assert_called_once_with(
        previous,
        previous,
        action=1,
        action_feasible=True,
        emptying_events=(),
        return_events=(),
        terminated=True,
        succeeded=False,
    )


# ======================================================================
# Lifecycle
# ======================================================================


def test_stop_is_idempotent_when_not_running() -> None:
    (
        environment,
        plant,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _make_environment()

    environment.stop()

    plant.stop_simulation.assert_not_called()


def test_stop_observes_event_controller() -> None:
    (
        environment,
        plant,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _make_environment()

    environment._running = True

    plant.read_event_controller_signals.side_effect = [
        (False, True, 4.0),
        (False, False, 5.0),
    ]

    running_observation = (
        environment.observe_event_controller()
    )

    assert (
        running_observation.state
        is EventControllerState.RUNNING
    )

    environment.stop()

    assert environment.running is False

    assert (
        environment.event_controller_observation
        is not None
    )

    assert (
        environment.event_controller_observation.state
        is EventControllerState.QUIESCENT
    )

    plant.stop_simulation.assert_called_once()


# ======================================================================
# Validation
# ======================================================================


def test_step_rejects_environment_without_reset() -> None:
    (
        environment,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _make_environment()

    log_probability, value = _neutral_tensors()

    with pytest.raises(
        RuntimeError,
        match="has not been reset",
    ):
        environment.step(
            0,
            log_probability,
            value,
        )


# ======================================================================
# Reset
# ======================================================================


def test_reset_clears_incomplete_tcp_handshake(
    monkeypatch,
) -> None:
    (
        environment,
        plant,
        _,
        observation_source,
        observation_encoder,
        _,
        emptying_event_reader,
        return_event_reader,
    ) = _make_environment()

    # Use a real TCPClient so that we test the actual handshake lifecycle.
    connection = MagicMock()
    tcp_client = TCPClient(connection)

    environment.tcp_client = tcp_client

    # Simulate the state left behind when an episode ends at a horizon.
    tcp_client._handshake.begin_decision(1000)

    assert (
        tcp_client.handshake.state
        is HandshakeState.WAITING_FOR_ACTION
    )

    initial_observation = ObservationMessage(
        decision_point_id=1,
        terminal=False,
    )

    connection.receive.return_value = (
        initial_observation
    )

    semantic_observation = _observation(
        simulation_time=1.0,
        throughput=0,
    )

    encoded_observation = torch.tensor([1.0])

    observation_source.read.return_value = (
        semantic_observation
    )

    observation_encoder.encode.return_value = (
        encoded_observation
    )

    # Keep this test focused on the TCP reset rather than the
    # EventController polling implementation.
    monkeypatch.setattr(
        environment,
        "_wait_for_event_controller_state",
        MagicMock(),
    )

    environment.reset()

    emptying_event_reader.reset.assert_called_once()
    return_event_reader.reset.assert_called_once()

    assert (
        tcp_client.handshake.state
        is HandshakeState.WAITING_FOR_ACTION
    )

    assert (
        tcp_client.handshake.decision_point_id
        == 1
    )

    assert (
        tcp_client.handshake.action_id
        is None
    )