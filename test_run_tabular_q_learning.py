from __future__ import annotations

from unittest.mock import Mock

import run_tabular_q_learning


def test_main_constructs_settings_and_runs_one_episode(
    monkeypatch,
):
    run_mock = Mock()

    monkeypatch.setattr(
        run_tabular_q_learning,
        "run",
        run_mock,
    )

    run_tabular_q_learning.main()

    run_mock.assert_called_once()

    _, kwargs = run_mock.call_args

    assert kwargs["num_episodes"] == 10

    assert kwargs["plant_settings"] is not None
    assert kwargs["tcp_settings"] is not None