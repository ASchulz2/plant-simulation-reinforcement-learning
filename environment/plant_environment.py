from __future__ import annotations

import time
from dataclasses import dataclass

from torch import Tensor

from problem.observation_source import ObservationSource
from environment.emptying_event_reader import EmptyingEventReader
from environment.rl_state_reset import (
    RLStateResetSettings,
    RLStateResetter,
)
from environment.event_controller_observer import (
    EventControllerObservation,
    EventControllerObserver,
    EventControllerState,
)
from environment.termination import TerminationReason
from environment.transition import Transition
from problem.observation import Observation
from problem.observation_encoder import ObservationEncoder
from problem.problem import ManufacturingProblem
from problem.reward import RewardFunction
from simulation.plant_simulation import PlantSimulation
from tcp.client import TCPClient
from tcp.connection import ConnectionError as TCPConnectionError
from tcp.protocol import (
    ObservationMessage,
    TerminalMessage,
)
from environment.return_event_reader import ReturnEventReader
from problem.tabular_state_encoder import (
    TabularStateEncoder,
)

@dataclass(frozen=True)
class PlantEnvironmentSettings:
    """Configuration of the Plant Simulation RL environment."""

    robot_path: str = ".Models.Model.PickAndPlaceRobot"

    max_steps: int = 1000

    poll_interval: float = 0.001
    lifecycle_timeout: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.max_steps, int):
            raise TypeError(
                "max_steps must be an integer."
            )

        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be positive."
            )

        if not isinstance(
            self.poll_interval,
            (int, float),
        ):
            raise TypeError(
                "poll_interval must be numeric."
            )

        if self.poll_interval <= 0.0:
            raise ValueError(
                "poll_interval must be positive."
            )

        if not isinstance(
            self.lifecycle_timeout,
            (int, float),
        ):
            raise TypeError(
                "lifecycle_timeout must be numeric."
            )

        if self.lifecycle_timeout <= 0.0:
            raise ValueError(
                "lifecycle_timeout must be positive."
            )


class PlantEnvironment:
    """
    Reinforcement-learning environment for Plant Simulation.

    Responsibilities
    ----------------
    - control Plant Simulation lifecycle
    - synchronize robot decisions through TCP
    - obtain semantic observations
    - obtain reward-relevant storage emptying events
    - encode observations for compatibility with existing RL code
    - calculate rewards
    - construct RL transitions
    - manage the RL episode horizon
    - classify episode termination

    Synchronization is exclusively performed through TCP for
    decision transactions.

    COM is used only for:
        - simulation lifecycle
        - physical state access through ObservationSource
        - reward-relevant emptying-event state
        - RL-state reset
        - EventController lifecycle observation

    EventController lifecycle
    -------------------------
        Reset
            ↓
        Ready
            ↓
        Running
            ↓
        Quiescent

    RL episode termination
    ----------------------
        success
        manufacturing_failure
        horizon_timeout
        communication_failure

    Decision transaction
    --------------------
        OBSERVATION
            ↓
        ACTION
            ↓
        ACK
            ↓
        next episode message
            ├── OBSERVATION
            └── TERMINAL
    """

    def __init__(
        self,
        plant: PlantSimulation,
        tcp_client: TCPClient,
        observation_source: ObservationSource,
        observation_encoder: ObservationEncoder,
        reward_calculator: RewardFunction,
        problem: ManufacturingProblem,
        event_controller_observer: EventControllerObserver,
        emptying_event_reader: EmptyingEventReader,
        return_event_reader: ReturnEventReader,
        settings: PlantEnvironmentSettings | None = None,
    ) -> None:
        self.plant = plant
        self.tcp_client = tcp_client
        self.observation_source = observation_source
        self.observation_encoder = observation_encoder
        self.reward_calculator = reward_calculator
        self.problem = problem
        self.tabular_state_encoder = (
            TabularStateEncoder(
                problem=problem,
            )
        )
        self.event_controller_observer = event_controller_observer
        self.emptying_event_reader = emptying_event_reader
        self.return_event_reader = return_event_reader

        self.settings = (
            settings
            if settings is not None
            else PlantEnvironmentSettings()
        )

        self._event_controller_observation: (
            EventControllerObservation | None
        ) = None

        self._current_decision_point_id: int | None = None

        self._current_semantic_observation: (
            Observation | None
        ) = None

        self._current_observation: Tensor | None = None

        self._last_action: int | None = None

        self._running: bool = False

        self._step_count: int = 0

        self._termination_reason: (
            TerminationReason | None
        ) = None

        # Compatibility with the existing Transition interface.
        self._progress_failure: bool = False
        self._successful_completion: bool = False

        self.rl_state_resetter = RLStateResetter(
            plant=plant,
            settings=RLStateResetSettings(
                robot_path=self.settings.robot_path,
            ),
        )

    # ==================================================================
    # EventController
    # ==================================================================

    def observe_event_controller(
        self,
    ) -> EventControllerObservation:
        """Read and interpret the current EventController signals."""

        (
            in_reset,
            is_running,
            sim_time,
        ) = self.plant.read_event_controller_signals()

        observation = self.event_controller_observer.observe(
            in_reset=in_reset,
            is_running=is_running,
            sim_time=sim_time,
        )

        self._event_controller_observation = observation

        return observation

    def _wait_for_event_controller_state(
        self,
        expected_state: EventControllerState,
    ) -> EventControllerObservation:
        start_time = time.monotonic()

        while True:
            observation = self.observe_event_controller()

            if observation.state == expected_state:
                return observation

            elapsed = time.monotonic() - start_time

            if elapsed >= self.settings.lifecycle_timeout:
                simulation_running = self.plant.is_simulation_running()
                simulation_error = self.plant.has_simulation_error()

                raise RuntimeError(
                    "Timed out waiting for Plant Simulation "
                    f"EventController state {expected_state.value}.\n"
                    "EventController:\n"
                    f"    InReset={observation.in_reset}\n"
                    f"    IsRunning={observation.is_running}\n"
                    f"    SimTime={observation.sim_time}\n"
                    f"    State={observation.state.value}\n"
                    "RemoteControl:\n"
                    f"    IsSimulationRunning={simulation_running}\n"
                    f"    HasSimulationError={simulation_error}"
                )

            time.sleep(self.settings.poll_interval)

    @property
    def event_controller_observation(
        self,
    ) -> EventControllerObservation | None:
        """Return the most recently observed EventController state."""

        return self._event_controller_observation

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self) -> Tensor:
        """
        Reset Plant Simulation and start a new RL episode.

        The reset establishes:

            Reset -> Ready -> Running

        The TCP connection is then expected to provide the first
        ObservationMessage for the new episode.

        The encoded tensor is returned for compatibility with the
        existing generic RL environment interface. The semantic
        Observation is simultaneously available through current_state.
        """

        # --------------------------------------------------------------
        # Reset local episode state.
        # --------------------------------------------------------------

        self._current_decision_point_id = None
        self._current_semantic_observation = None
        self._current_observation = None
        self._last_action = None
        self._running = False
        self._step_count = 0
        self._termination_reason = None
        self._progress_failure = False
        self._successful_completion = False

        self.emptying_event_reader.reset()
        self.return_event_reader.reset()

        self.tcp_client.reset_handshake()

        print("Resetting environment...")

        # The reset itself may restore RLConnection.on,
        # so do not enable the connection before this.
        self.plant.disable_rl_connection()
        self.plant.reset_simulation()

        self._wait_for_event_controller_state(EventControllerState.READY)

        self._configure_problem_parameters()
        self.rl_state_resetter.reset()

        # Enable only after ResetSimulation().
        self.plant.enable_rl_connection()

        # Re-establish the Python TCP connection against
        # the freshly enabled Plant Simulation server.
        if self.tcp_client.can_reconnect:
            self.tcp_client.reconnect()
        else:
            self.tcp_client.reset_handshake()

        print("Starting Plant Simulation...")

        self.plant.start_simulation()

        self._wait_for_event_controller_state(EventControllerState.RUNNING)

        self._running = True

        # --------------------------------------------------------------
        # Wait for the first RL observation.
        # --------------------------------------------------------------

        try:
            observation_message = (
                self.tcp_client.receive_observation()
            )
        except TCPConnectionError as exc:
            self._terminate_for_communication_failure(
                str(exc)
            )

            raise RuntimeError(
                "Failed to receive the initial "
                "Plant Simulation observation."
            ) from exc

        # --------------------------------------------------------------
        # Validate the initial observation.
        # --------------------------------------------------------------

        if observation_message.decision_point_id < 1:
            self._terminate_for_communication_failure(
                "Invalid initial decision-point ID: "
                f"{observation_message.decision_point_id}."
            )

            raise RuntimeError(
                "Plant Simulation sent an invalid initial "
                "decision-point ID: "
                f"{observation_message.decision_point_id}."
            )

        if observation_message.terminal:
            self._terminate_for_communication_failure(
                "Initial ObservationMessage unexpectedly "
                "contained terminal=True."
            )

            raise RuntimeError(
                "Plant Simulation sent a terminal observation "
                "while resetting the environment."
            )

        self._current_decision_point_id = (
            observation_message.decision_point_id
        )

        # --------------------------------------------------------------
        # Read semantic observation.
        # --------------------------------------------------------------

        semantic_observation = (
            self.observation_source.read()
        )

        encoded_observation = (
            self.observation_encoder.encode(
                semantic_observation
            )
        )

        self._current_semantic_observation = (
            semantic_observation
        )

        self._current_observation = (
            encoded_observation
        )

        print("Environment ready.")

        return encoded_observation

    # ==================================================================
    # Step
    # ==================================================================

    def step(
        self,
        action: int,
        log_probability: Tensor,
        value: Tensor,
    ) -> Transition:
        """
        Execute one complete RL decision.

        The Plant Simulation sequence is:

            ACTION
              ↓
            ACK
              ↓
            next OBSERVATION or TERMINAL

        A normal observation may become a horizon terminal after
        the transition has been processed.
        """

        # --------------------------------------------------------------
        # Environment state validation.
        # --------------------------------------------------------------

        if self._current_semantic_observation is None:
            raise RuntimeError(
                "Environment has not been reset."
            )

        if self._current_observation is None:
            raise RuntimeError(
                "Current observation is unavailable."
            )

        if self._current_decision_point_id is None:
            raise RuntimeError(
                "Current decision-point ID is unavailable."
            )

        if not self._running:
            raise RuntimeError(
                "Plant Simulation is not running."
            )

        # --------------------------------------------------------------
        # Validate action through the problem definition.
        # --------------------------------------------------------------

        self.problem.validate_action(action)

        previous_semantic_observation = (
            self._current_semantic_observation
        )

        previous_observation = self._current_observation

        decision_point_id = self._current_decision_point_id

        self._last_action = action

        action_feasible = self._is_action_feasible(
            previous_semantic_observation,
            action,
        )

        # --------------------------------------------------------------
        # ACTION -> ACK -> next message
        # --------------------------------------------------------------

        try:
            self.tcp_client.send_action(
                decision_point_id=decision_point_id,
                action_id=action,
            )

            acknowledgement = (
                self.tcp_client.receive_acknowledgement()
            )

            if (
                acknowledgement.decision_point_id
                != decision_point_id
            ):
                raise TCPConnectionError(
                    "Acknowledgement decision-point mismatch: "
                    f"expected {decision_point_id}, "
                    f"received "
                    f"{acknowledgement.decision_point_id}."
                )

            self.tcp_client.complete_handshake()

            message = (
                self.tcp_client.receive_episode_message()
            )

        except TCPConnectionError as exc:
            return self._build_communication_failure_transition(
                previous_semantic_observation=(
                    previous_semantic_observation
                ),
                previous_observation=previous_observation,
                action=action,
                log_probability=log_probability,
                value=value,
                reason=str(exc),
            )

        # --------------------------------------------------------------
        # Explicit terminal message.
        # --------------------------------------------------------------

        if isinstance(message, TerminalMessage):
            return self._handle_terminal_message(
                message=message,
                previous_semantic_observation=(
                    previous_semantic_observation
                ),
                previous_observation=previous_observation,
                action=action,
                log_probability=log_probability,
                value=value,
            )

        # --------------------------------------------------------------
        # Unexpected message type.
        # --------------------------------------------------------------

        if not isinstance(message, ObservationMessage):
            return self._build_communication_failure_transition(
                previous_semantic_observation=(
                    previous_semantic_observation
                ),
                previous_observation=previous_observation,
                action=action,
                log_probability=log_probability,
                value=value,
                reason=(
                    "Unexpected episode message: "
                    f"{type(message).__name__}."
                ),
            )

        # --------------------------------------------------------------
        # Observation validation.
        # --------------------------------------------------------------

        if message.terminal:
            return self._build_communication_failure_transition(
                previous_semantic_observation=(
                    previous_semantic_observation
                ),
                previous_observation=previous_observation,
                action=action,
                log_probability=log_probability,
                value=value,
                reason=(
                    "ObservationMessage unexpectedly contained "
                    "terminal=True."
                ),
            )

        next_decision_point_id = message.decision_point_id

        if next_decision_point_id <= decision_point_id:
            return self._build_communication_failure_transition(
                previous_semantic_observation=(
                    previous_semantic_observation
                ),
                previous_observation=previous_observation,
                action=action,
                log_probability=log_probability,
                value=value,
                reason=(
                    "Decision-point ID did not increase: "
                    f"{decision_point_id} -> "
                    f"{next_decision_point_id}."
                ),
            )

        self._current_decision_point_id = (
            next_decision_point_id
        )

        # --------------------------------------------------------------
        # Read next semantic observation.
        # --------------------------------------------------------------

        next_semantic_observation = (
            self.observation_source.read()
        )

        # --------------------------------------------------------------
        # Read reward-relevant events.
        # --------------------------------------------------------------

        emptying_events = (
            self.emptying_event_reader.read()
        )

        return_events = (
            self.return_event_reader.read()
        )

        # --------------------------------------------------------------
        # Count the completed RL step.
        # --------------------------------------------------------------

        self._step_count += 1

        # --------------------------------------------------------------
        # Determine terminal condition.
        # --------------------------------------------------------------

        terminated = False
        successful_completion = False
        termination_reason: TerminationReason | None = None

        if self._reached_horizon():
            terminated = True
            termination_reason = (
                TerminationReason.HORIZON_TIMEOUT
            )

        # --------------------------------------------------------------
        # Reward.
        # --------------------------------------------------------------

        reward = self.reward_calculator.calculate(
            previous_semantic_observation,
            next_semantic_observation,
            action=action,
            action_feasible=action_feasible,
            emptying_events=emptying_events,
            return_events=return_events,
            terminated=terminated,
            succeeded=successful_completion,
        )

        # --------------------------------------------------------------
        # Elapsed simulation time.
        # --------------------------------------------------------------

        elapsed_time = (
            next_semantic_observation.simulation_time
            - previous_semantic_observation.simulation_time
        )

        if elapsed_time < 0.0:
            raise RuntimeError(
                "Simulation time decreased between "
                "two consecutive decision points: "
                f"{previous_semantic_observation.simulation_time}"
                " -> "
                f"{next_semantic_observation.simulation_time}"
            )

        # --------------------------------------------------------------
        # Encode next observation.
        # --------------------------------------------------------------

        next_observation = (
            self.observation_encoder.encode(
                next_semantic_observation
            )
        )

        self._current_semantic_observation = (
            next_semantic_observation
        )

        self._current_observation = next_observation

        self._termination_reason = termination_reason

        self._progress_failure = False
        self._successful_completion = (
            successful_completion
        )

        # --------------------------------------------------------------
        # Stop after horizon termination.
        # --------------------------------------------------------------

        if terminated:
            self._running = False

            self._stop_simulation_safely()

            self.observe_event_controller()

        return Transition(
            state=previous_observation,
            action=action,
            reward=reward,
            next_state=next_observation,
            log_probability=log_probability.detach(),
            value=value.detach(),
            terminated=terminated,
            elapsed_time=elapsed_time,
            simulation_time=(
                previous_semantic_observation.simulation_time
            ),
            next_simulation_time=(
                next_semantic_observation.simulation_time
            ),
            throughput=(
                previous_semantic_observation.throughput
            ),
            next_throughput=(
                next_semantic_observation.throughput
            ),
            progress_failure=False,
            succeeded=successful_completion,
            termination_reason=termination_reason,
            emptying_events=emptying_events
        )

    # ==================================================================
    # Terminal handling
    # ==================================================================

    def _handle_terminal_message(
        self,
        *,
        message: TerminalMessage,
        previous_semantic_observation: Observation,
        previous_observation: Tensor,
        action: int,
        log_probability: Tensor,
        value: Tensor,
    ) -> Transition:
        """
        Convert a Plant Simulation terminal message
        into a terminal transition.
        """

        if (
            message.decision_point_id
            != self._current_decision_point_id
        ):
            return self._build_communication_failure_transition(
                previous_semantic_observation=(
                    previous_semantic_observation
                ),
                previous_observation=previous_observation,
                action=action,
                log_probability=log_probability,
                value=value,
                reason=(
                    "Terminal decision-point mismatch: "
                    f"expected "
                    f"{self._current_decision_point_id}, "
                    f"received "
                    f"{message.decision_point_id}."
                ),
            )

        reason = message.reason.strip().lower()

        if reason == "success":
            successful_completion = True
            termination_reason = (
                TerminationReason.SUCCESS
            )

        elif reason == "failure":
            successful_completion = False
            termination_reason = (
                TerminationReason.MANUFACTURING_FAILURE
            )

        else:
            return self._build_communication_failure_transition(
                previous_semantic_observation=(
                    previous_semantic_observation
                ),
                previous_observation=previous_observation,
                action=action,
                log_probability=log_probability,
                value=value,
                reason=(
                    "Unknown terminal reason: "
                    f"{message.reason!r}."
                ),
            )

        # --------------------------------------------------------------
        # Read final semantic observation.
        # --------------------------------------------------------------

        next_semantic_observation = (
            self.observation_source.read()
        )

        # --------------------------------------------------------------
        # Read final reward-relevant events.
        # --------------------------------------------------------------

        emptying_events = (
            self.emptying_event_reader.read()
        )

        return_events = (
            self.return_event_reader.read()
        )

        action_feasible = self._is_action_feasible(
            previous_semantic_observation,
            action,
        )

        # --------------------------------------------------------------
        # Reward.
        # --------------------------------------------------------------

        reward = self.reward_calculator.calculate(
            previous_semantic_observation,
            next_semantic_observation,
            action=action,
            action_feasible=action_feasible,
            emptying_events=emptying_events,
            return_events=return_events,
            terminated=True,
            succeeded=successful_completion,
        )

        # --------------------------------------------------------------
        # Elapsed simulation time.
        # --------------------------------------------------------------

        elapsed_time = (
            next_semantic_observation.simulation_time
            - previous_semantic_observation.simulation_time
        )

        if elapsed_time < 0.0:
            raise RuntimeError(
                "Simulation time decreased between "
                "two consecutive states: "
                f"{previous_semantic_observation.simulation_time}"
                " -> "
                f"{next_semantic_observation.simulation_time}"
            )

        # --------------------------------------------------------------
        # Encode next observation.
        # --------------------------------------------------------------

        next_observation = (
            self.observation_encoder.encode(
                next_semantic_observation
            )
        )

        self._current_semantic_observation = (
            next_semantic_observation
        )

        self._current_observation = next_observation

        self._termination_reason = termination_reason

        self._progress_failure = False
        self._successful_completion = (
            successful_completion
        )

        self._running = False

        self._stop_simulation_safely()

        self.observe_event_controller()

        return Transition(
            state=previous_observation,
            action=action,
            reward=reward,
            next_state=next_observation,
            log_probability=log_probability.detach(),
            value=value.detach(),
            terminated=True,
            elapsed_time=elapsed_time,
            throughput=(
                previous_semantic_observation.throughput
            ),
            next_throughput=(
                next_semantic_observation.throughput
            ),
            simulation_time=(
                previous_semantic_observation.simulation_time
            ),
            next_simulation_time=(
                next_semantic_observation.simulation_time
            ),
            progress_failure=False,
            succeeded=successful_completion,
            termination_reason=termination_reason,
            emptying_events=emptying_events
        )

    # ==================================================================
    # Communication failure
    # ==================================================================

    def _terminate_for_communication_failure(
        self,
        reason: str,
    ) -> None:
        """Mark the environment as terminated due to communication failure."""

        self._progress_failure = True
        self._successful_completion = False

        self._termination_reason = (
            TerminationReason.COMMUNICATION_FAILURE
        )

        self._running = False

        print(
            "TCP communication failure:",
            reason,
        )

        self._stop_simulation_safely()

    def _build_communication_failure_transition(
        self,
        *,
        previous_semantic_observation: Observation,
        previous_observation: Tensor,
        action: int,
        log_probability: Tensor,
        value: Tensor,
        reason: str,
    ) -> Transition:
        """
        Build a terminal transition for communication failure.

        No new physical observation is assumed to exist. The last known
        semantic and encoded observations are therefore used for both
        sides of the transition.

        No emptying event is read here because no new physical
        decision-point observation exists.
        """

        self._terminate_for_communication_failure(reason)
        action_feasible = self._is_action_feasible(
            previous_semantic_observation,
            action,
        )

        reward = self.reward_calculator.calculate(
            previous_semantic_observation,
            previous_semantic_observation,
            action=action,
            action_feasible=action_feasible,
            emptying_events=(),
            return_events=(),
            terminated=True,
            succeeded=False,
        )

        return Transition(
            state=previous_observation,
            action=action,
            reward=reward,
            next_state=previous_observation,
            log_probability=log_probability.detach(),
            value=value.detach(),
            terminated=True,
            elapsed_time=0.0,
            throughput=(
                previous_semantic_observation.throughput
            ),
            next_throughput=(
                previous_semantic_observation.throughput
            ),
            simulation_time=(
                previous_semantic_observation.simulation_time
            ),
            next_simulation_time=(
                previous_semantic_observation.simulation_time
            ),
            progress_failure=True,
            succeeded=False,
            termination_reason=(
                TerminationReason.COMMUNICATION_FAILURE
            ),
            emptying_events=(),
        )

    # ==================================================================
    # Horizon
    # ==================================================================

    def _reached_horizon(self) -> bool:
        """Return whether the RL episode reached its step horizon."""

        return (
            self._step_count
            >= self.settings.max_steps
        )

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def stop(self) -> None:
        """
        Stop Plant Simulation and mark the environment inactive.

        A normal explicit stop does not itself assign an RL terminal
        reason. Terminal reasons are assigned by episode logic.
        """

        if not self._running:
            return

        self._running = False

        self._stop_simulation_safely()

        self.observe_event_controller()

    def _stop_simulation_safely(self) -> None:
        """Stop simulation without masking an earlier failure."""

        try:
            self.plant.stop_simulation()
        except Exception as exc:
            print(
                "Warning: failed to stop Plant Simulation:",
                exc,
            )

    def _configure_problem_parameters(self) -> None:
        self.plant.set_value(
            ".Models.Model.AdmissionBuffer.Capacity",
            self.problem.admission_buffer_capacity,
        )
        self.plant.set_value(
            ".Models.Model.StorageA.Capacity",
            self.problem.storage_capacity,
        )
        self.plant.set_value(
            ".Models.Model.StorageB.Capacity",
            self.problem.storage_capacity,
        )
        self.plant.set_value(
            ".Models.Model.DrainA.RL_targetThroughput",
            self.problem.target_throughput,
        )

    def _is_action_feasible(
        self,
        observation: Observation,
        action: int,
    ) -> bool:
        """Return whether an action is feasible in the given state."""

        tabular_state = (
            self.tabular_state_encoder.encode(
                observation
            )
        )

        feasible_actions = (
            self.problem.feasible_actions(
                tabular_state
            )
        )

        return action in feasible_actions

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def running(self) -> bool:
        return self._running

    @property
    def current_state(
        self,
    ) -> Observation | None:
        """
        Return the current semantic observation.

        This is the state consumed by the tabular agent.
        """

        return self._current_semantic_observation

    @property
    def current_observation(
        self,
    ) -> Tensor | None:
        """
        Return the encoded tensor observation.

        This remains available for the generic RL interface.
        """

        return self._current_observation

    @property
    def decision_point_id(
        self,
    ) -> int | None:
        return self._current_decision_point_id

    @property
    def current_decision_point_id(
        self,
    ) -> int | None:
        """Return the current RL decision-point identifier."""

        return self._current_decision_point_id

    @property
    def action(
        self,
    ) -> int | None:
        return self._last_action

    @property
    def step_count(self) -> int:
        """Return the number of completed environment steps."""

        return self._step_count

    @property
    def termination_reason(
        self,
    ) -> TerminationReason | None:
        """Return the reason for the most recent termination."""

        return self._termination_reason

    @property
    def progress_failure(self) -> bool:
        """
        Compatibility property.

        This remains true only for communication failures.
        """

        return self._progress_failure

    @property
    def successful_completion(self) -> bool:
        """Return whether the current episode ended successfully."""

        return self._successful_completion