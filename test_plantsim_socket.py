from __future__ import annotations

from config.tcp_settings import TCPSettings
from tcp.client import TCPClient
from tcp.handshake import HandshakeState


def run_decision_cycle(
    client: TCPClient,
    *,
    cycle_number: int,
    action_id: int,
    expected_decision_point_id: int | None = None,
) -> int:

    print(f"\n=== DECISION CYCLE {cycle_number} ===")

    print("Waiting for robot OBSERVATION...")
    observation = client.receive_observation()

    print("Received OBSERVATION:", observation)

    assert observation.terminal is False
    assert observation.state is None

    decision_point_id = observation.decision_point_id

    if expected_decision_point_id is not None:
        assert decision_point_id == expected_decision_point_id

    assert (
        client.handshake.state
        is HandshakeState.WAITING_FOR_ACTION
    )

    print(
        "Sending ACTION:",
        f"decision_point_id={decision_point_id}, "
        f"action_id={action_id}",
    )

    client.send_action(
        decision_point_id=decision_point_id,
        action_id=action_id,
    )

    assert (
        client.handshake.state
        is HandshakeState.ACTION_SENT
    )

    print("Waiting for ACK...")
    acknowledgement = client.receive_acknowledgement()

    print("Received ACK:", acknowledgement)

    assert (
        acknowledgement.decision_point_id
        == decision_point_id
    )

    assert (
        client.handshake.state
        is HandshakeState.ACKNOWLEDGED
    )

    client.complete_handshake()

    assert (
        client.handshake.state
        is HandshakeState.IDLE
    )

    print(
        f"Decision cycle {cycle_number} completed."
    )

    return decision_point_id


def main() -> None:

    settings = TCPSettings(
        host="127.0.0.1",
        port=5000,
        receive_timeout=60.0,
    )

    print("Connecting to Plant Simulation...")

    client = TCPClient.connect(
        settings,
        protocol_version="1.0",
    )

    print("Connected.")

    try:
        # ----------------------------------------------------------
        # Connection-level handshake
        # ----------------------------------------------------------

        print("Sending HELLO...")
        client.send_hello()

        print("Waiting for HELLO_ACK...")
        hello_ack = client.receive_hello_ack()

        assert hello_ack.protocol_version == "1.0"

        print("Protocol handshake successful.")

        # ----------------------------------------------------------
        # Three physical decision cycles
        # ----------------------------------------------------------

        decision_point_id = run_decision_cycle(
            client,
            cycle_number=1,
            action_id=0,
            expected_decision_point_id=1,
        )

        decision_point_id = run_decision_cycle(
            client,
            cycle_number=2,
            action_id=1,
            expected_decision_point_id=decision_point_id + 1,
        )

        decision_point_id = run_decision_cycle(
            client,
            cycle_number=3,
            action_id=2,
            expected_decision_point_id=decision_point_id + 1,
        )

        print(
            "\nAll three decision cycles "
            "completed successfully."
        )

    finally:
        client.close()

    print("Connection closed.")


if __name__ == "__main__":
    main()