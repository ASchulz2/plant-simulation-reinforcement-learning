from __future__ import annotations

from dataclasses import dataclass, field

from config.tcp_settings import TCPSettings
from tcp.client import TCPClient
from tcp.connection import ConnectionError as TCPConnectionError
from tcp.handshake import HandshakeState
from tcp.protocol import (
    ObservationMessage,
    TerminalMessage,
)


@dataclass
class EpisodeDiagnostics:
    """Diagnostics accumulated during one episode."""

    episode_number: int

    decision_points: int = 0

    first_decision_point_id: int | None = None
    last_decision_point_id: int | None = None

    actions: list[int] = field(
        default_factory=list
    )

    terminal_id: int | None = None
    terminal_reason: str | None = None


def select_experimental_action(
    diagnostics: EpisodeDiagnostics,
) -> int:
    """
    Temporary deterministic policy.

    Cycles through the three physical destinations:

        0 -> STORAGE_A
        1 -> STORAGE_B
        2 -> RETURN_BUFFER

    This will later be replaced by the RL agent.
    """

    return diagnostics.decision_points % 3


def run_episode(
    client: TCPClient,
    *,
    episode_number: int,
) -> EpisodeDiagnostics:

    diagnostics = EpisodeDiagnostics(
        episode_number=episode_number
    )

    print("\n" + "=" * 70)
    print(f"EPISODE {episode_number}")
    print("=" * 70)

    while True:

        print("\nWaiting for episode event...")

        event = client.receive_episode_event()

        # ------------------------------------------------------
        # Explicit episode termination
        # ------------------------------------------------------

        if isinstance(
            event,
            TerminalMessage,
        ):

            diagnostics.terminal_id = (
                event.decision_point_id
            )

            diagnostics.terminal_reason = (
                event.reason
            )

            print("\nReceived TERMINAL:")
            print(event)

            break

        # ------------------------------------------------------
        # Decision point
        # ------------------------------------------------------

        observation: ObservationMessage = event

        print("\nReceived OBSERVATION:")
        print(observation)

        assert observation.terminal is False

        decision_point_id = (
            observation.decision_point_id
        )

        diagnostics.decision_points += 1

        if (
            diagnostics.first_decision_point_id
            is None
        ):
            diagnostics.first_decision_point_id = (
                decision_point_id
            )

        diagnostics.last_decision_point_id = (
            decision_point_id
        )

        assert (
            client.handshake.state
            is HandshakeState.WAITING_FOR_ACTION
        )

        # ------------------------------------------------------
        # Experimental action
        # ------------------------------------------------------

        action_id = select_experimental_action(
            diagnostics
        )

        diagnostics.actions.append(
            action_id
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

        # ------------------------------------------------------
        # ACK
        # ------------------------------------------------------

        print("Waiting for ACK...")

        acknowledgement = (
            client.receive_acknowledgement()
        )

        print(
            "Received ACK:",
            acknowledgement,
        )

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


    return diagnostics


def print_episode_diagnostics(
    diagnostics: EpisodeDiagnostics,
) -> None:

    print("\n" + "-" * 70)
    print(
        f"EPISODE {diagnostics.episode_number} "
        "DIAGNOSTICS"
    )
    print("-" * 70)

    print(
        "Decision points:",
        diagnostics.decision_points,
    )

    print(
        "First decision point:",
        diagnostics.first_decision_point_id,
    )

    print(
        "Last decision point:",
        diagnostics.last_decision_point_id,
    )

    print(
        "Actions:",
        diagnostics.actions,
    )

    print(
        "Terminal ID:",
        diagnostics.terminal_id,
    )

    print(
        "Terminal reason:",
        diagnostics.terminal_reason,
    )


def ask_for_next_episode() -> bool:
    answer = input(
        "\nStart another episode? [Y/n]: "
    ).strip().lower()

    return answer in {
        "",
        "y",
        "yes",
    }


def main() -> None:

    settings = TCPSettings(
        host="127.0.0.1",
        port=5000,
        receive_timeout=60.0,
    )

    print(
        "Connecting to Plant Simulation..."
    )

    client = TCPClient.connect(
        settings,
        protocol_version="1.0",
    )

    print("Connected.")

    try:

        # ------------------------------------------------------
        # Connection-level handshake
        # ------------------------------------------------------

        print("Sending HELLO...")

        client.send_hello()

        print("Waiting for HELLO_ACK...")

        hello_ack = client.receive_hello_ack()

        assert (
            hello_ack.protocol_version
            == "1.0"
        )

        print(
            "Protocol handshake successful."
        )

        # ------------------------------------------------------
        # Interactive episode loop
        # ------------------------------------------------------

        episode_number = 1

        while True:

            try:

                diagnostics = run_episode(
                    client,
                    episode_number=episode_number,
                )

            except TCPConnectionError as exc:

                print(
                    "\nTCP connection failure:"
                )
                print(exc)

                break

            print_episode_diagnostics(
                diagnostics
            )

            if not ask_for_next_episode():
                break

            episode_number += 1

    except KeyboardInterrupt:

        print(
            "\nExperiment interrupted by user."
        )

    finally:

        client.close()

    print(
        "\nConnection closed."
    )


if __name__ == "__main__":
    main()