import pytest

from config.tcp_settings import TCPSettings


def test_default_settings() -> None:
    settings = TCPSettings()

    assert settings.host == "127.0.0.1"
    assert settings.port == 5000
    assert settings.connect_timeout == 5.0
    assert settings.receive_timeout == 60.0
    assert settings.backlog == 1


def test_invalid_port() -> None:
    with pytest.raises(ValueError):
        TCPSettings(port=70000)


def test_invalid_timeout() -> None:
    with pytest.raises(ValueError):
        TCPSettings(connect_timeout=0.0)


def test_invalid_backlog() -> None:
    with pytest.raises(ValueError):
        TCPSettings(backlog=0)