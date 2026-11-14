import pytest

from problem.q_learning import (
    QLearningSettings,
    QLearningUpdater,
)
from problem.q_table import QTable
from problem.tabular_state import TabularState


def make_state(
    current_language: str = "GER",
) -> TabularState:
    return TabularState(
        current_language=current_language,
        admission_buffer_ger_count=1,
        admission_buffer_fr_count=0,
        admission_buffer_rus_count=0,
        storage_a_language="GER",
        storage_a_count=0,
        storage_b_language="FR",
        storage_b_count=0,
    )


def test_non_terminal_q_learning_update():
    table = QTable()

    state = make_state("GER")
    next_state = make_state("FR")

    table.set(
        next_state,
        0,
        4.0,
    )
    table.set(
        next_state,
        1,
        2.0,
    )
    table.set(
        next_state,
        2,
        3.0,
    )

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(
            learning_rate=0.5,
            discount_factor=0.9,
        ),
    )

    updated_value = updater.update(
        state=state,
        action=1,
        reward=2.0,
        next_state=next_state,
        terminated=False,
        actions=(0, 1, 2),
    )

    expected_target = 2.0 + 0.9 * 4.0
    expected_value = 0.5 * expected_target

    assert updated_value == pytest.approx(
        expected_value
    )

    assert table.get(
        state,
        1,
    ) == pytest.approx(
        expected_value
    )


def test_terminal_update_does_not_bootstrap():
    table = QTable()

    state = make_state("GER")
    next_state = make_state("FR")

    table.set(
        next_state,
        0,
        100.0,
    )

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(
            learning_rate=0.5,
            discount_factor=0.9,
        ),
    )

    updated_value = updater.update(
        state=state,
        action=1,
        reward=10.0,
        next_state=next_state,
        terminated=True,
        actions=(0, 1, 2),
    )

    assert updated_value == pytest.approx(5.0)


def test_existing_q_value_is_updated_incrementally():
    table = QTable()

    state = make_state()
    next_state = make_state("RUS")

    table.set(
        state,
        0,
        4.0,
    )

    table.set(
        next_state,
        0,
        8.0,
    )

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(
            learning_rate=0.25,
            discount_factor=0.5,
        ),
    )

    updated_value = updater.update(
        state=state,
        action=0,
        reward=2.0,
        next_state=next_state,
        terminated=False,
        actions=(0,),
    )

    target = 2.0 + 0.5 * 8.0
    expected = 4.0 + 0.25 * (target - 4.0)

    assert updated_value == pytest.approx(
        expected
    )


def test_non_terminal_update_requires_actions():
    table = QTable()

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(),
    )

    with pytest.raises(ValueError):
        updater.update(
            state=make_state(),
            action=0,
            reward=1.0,
            next_state=make_state("FR"),
            terminated=False,
            actions=(),
        )


def test_terminal_update_does_not_require_actions():
    table = QTable()

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(),
    )

    updated_value = updater.update(
        state=make_state(),
        action=0,
        reward=3.0,
        next_state=make_state("FR"),
        terminated=True,
        actions=(),
    )

    assert updated_value == pytest.approx(0.3)


@pytest.mark.parametrize(
    "learning_rate",
    [
        0.0,
        -0.1,
        1.1,
    ],
)
def test_learning_rate_is_validated(
    learning_rate: float,
):
    with pytest.raises(ValueError):
        QLearningSettings(
            learning_rate=learning_rate
        )


@pytest.mark.parametrize(
    "discount_factor",
    [
        -0.1,
        1.1,
    ],
)
def test_discount_factor_is_validated(
    discount_factor: float,
):
    with pytest.raises(ValueError):
        QLearningSettings(
            discount_factor=discount_factor
        )