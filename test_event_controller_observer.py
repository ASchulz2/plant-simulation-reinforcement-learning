from environment.event_controller_observer import (
    EventControllerObserver,
    EventControllerState,
)


def test_initial_state_is_reset() -> None:
    observer = EventControllerObserver()

    assert observer.state == EventControllerState.RESET


def test_reset_signal_produces_reset_state() -> None:
    observer = EventControllerObserver()

    observation = observer.observe(
        in_reset=True,
        is_running=False,
        sim_time=0.0,
    )

    assert observation.state == EventControllerState.RESET
    assert observer.state == EventControllerState.RESET


def test_reset_completion_produces_ready_state() -> None:
    observer = EventControllerObserver()

    observer.observe(
        in_reset=True,
        is_running=False,
        sim_time=0.0,
    )

    observation = observer.observe(
        in_reset=False,
        is_running=False,
        sim_time=0.0,
    )

    assert observation.state == EventControllerState.READY
    assert observer.state == EventControllerState.READY


def test_start_produces_running_state() -> None:
    observer = EventControllerObserver()

    observer.observe(
        in_reset=False,
        is_running=False,
        sim_time=0.0,
    )

    observation = observer.observe(
        in_reset=False,
        is_running=True,
        sim_time=1.5,
    )

    assert observation.state == EventControllerState.RUNNING
    assert observer.state == EventControllerState.RUNNING


def test_stop_after_running_produces_quiescent_state() -> None:
    observer = EventControllerObserver()

    observer.observe(
        in_reset=False,
        is_running=True,
        sim_time=10.0,
    )

    observation = observer.observe(
        in_reset=False,
        is_running=False,
        sim_time=10.0,
    )

    assert observation.state == EventControllerState.QUIESCENT
    assert observer.state == EventControllerState.QUIESCENT


def test_quiescent_state_remains_quiescent_while_stopped() -> None:
    observer = EventControllerObserver()

    observer.observe(
        in_reset=False,
        is_running=True,
        sim_time=10.0,
    )
    observer.observe(
        in_reset=False,
        is_running=False,
        sim_time=10.0,
    )

    observation = observer.observe(
        in_reset=False,
        is_running=False,
        sim_time=10.0,
    )

    assert observation.state == EventControllerState.QUIESCENT


def test_quiescent_state_can_return_to_running() -> None:
    observer = EventControllerObserver()

    observer.observe(
        in_reset=False,
        is_running=True,
        sim_time=10.0,
    )
    observer.observe(
        in_reset=False,
        is_running=False,
        sim_time=10.0,
    )

    observation = observer.observe(
        in_reset=False,
        is_running=True,
        sim_time=12.0,
    )

    assert observation.state == EventControllerState.RUNNING


def test_observation_preserves_simulation_time() -> None:
    observer = EventControllerObserver()

    observation = observer.observe(
        in_reset=False,
        is_running=True,
        sim_time=42.75,
    )

    assert observation.sim_time == 42.75


def test_integer_simulation_time_is_converted_to_float() -> None:
    observer = EventControllerObserver()

    observation = observer.observe(
        in_reset=False,
        is_running=True,
        sim_time=42,
    )

    assert observation.sim_time == 42.0


def test_negative_simulation_time_is_rejected() -> None:
    observer = EventControllerObserver()

    try:
        observer.observe(
            in_reset=False,
            is_running=False,
            sim_time=-1.0,
        )
    except ValueError as exc:
        assert str(exc) == "sim_time must be non-negative"
    else:
        raise AssertionError("Expected ValueError")


def test_non_boolean_reset_signal_is_rejected() -> None:
    observer = EventControllerObserver()

    try:
        observer.observe(
            in_reset=1,
            is_running=False,
            sim_time=0.0,
        )
    except TypeError as exc:
        assert str(exc) == "in_reset must be a bool"
    else:
        raise AssertionError("Expected TypeError")


def test_non_boolean_running_signal_is_rejected() -> None:
    observer = EventControllerObserver()

    try:
        observer.observe(
            in_reset=False,
            is_running=1,
            sim_time=0.0,
        )
    except TypeError as exc:
        assert str(exc) == "is_running must be a bool"
    else:
        raise AssertionError("Expected TypeError")


def test_non_numeric_simulation_time_is_rejected() -> None:
    observer = EventControllerObserver()

    try:
        observer.observe(
            in_reset=False,
            is_running=False,
            sim_time="0.0",
        )
    except TypeError as exc:
        assert str(exc) == "sim_time must be numeric"
    else:
        raise AssertionError("Expected TypeError")

def test_ready_can_be_detected_without_observing_reset_phase():
    observer = EventControllerObserver()

    observation = observer.observe(
        in_reset=False,
        is_running=False,
        sim_time=0.0,
    )

    assert observation.state == EventControllerState.READY

def test_reset_from_quiescent_can_reach_ready_without_reset_pulse():
    observer = EventControllerObserver()

    observer.observe(
        in_reset=False,
        is_running=True,
        sim_time=10.0,
    )

    observation = observer.observe(
        in_reset=False,
        is_running=False,
        sim_time=0.0,
    )

    assert observation.state == EventControllerState.READY