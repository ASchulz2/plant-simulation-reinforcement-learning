from config.tcp_settings import TCPSettings
from tcp.connection import TCPConnection


def main() -> None:
    settings = TCPSettings()

    print(f"Connecting to {settings.host}:{settings.port}...")

    connection = TCPConnection.connect(settings)

    print("TCP connection established.")
    print("Closing TCP connection...")

    connection.close()

    print("TCP connection closed.")


if __name__ == "__main__":
    main()