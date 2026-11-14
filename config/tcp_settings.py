from dataclasses import dataclass


@dataclass(frozen=True)
class TCPSettings:
    """Configuration for the TCP communication layer."""

    host: str = "127.0.0.1"
    port: int = 5000
    connect_timeout: float = 5.0
    receive_timeout: float = 60.0
    backlog: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError(
                "port must be between 0 and 65535."
            )

        if self.connect_timeout <= 0:
            raise ValueError(
                "connect_timeout must be positive."
            )

        if self.receive_timeout <= 0:
            raise ValueError(
                "receive_timeout must be positive."
            )

        if self.backlog <= 0:
            raise ValueError(
                "backlog must be positive."
            )