import pytest

from problem.action_space import ActionSpace


def test_default_action_space():
    action_space = ActionSpace()

    assert action_space.actions == (0, 1, 2)


@pytest.mark.parametrize(
    "action",
    [0, 1, 2],
)
def test_valid_actions_are_contained(
    action: int,
):
    action_space = ActionSpace()

    assert action_space.contains(action)


def test_unknown_action_is_not_contained():
    action_space = ActionSpace()

    assert not action_space.contains(3)


def test_validation_accepts_valid_action():
    action_space = ActionSpace()

    action_space.validate(1)


def test_validation_rejects_invalid_action():
    action_space = ActionSpace()

    with pytest.raises(ValueError):
        action_space.validate(3)


def test_validation_rejects_non_integer_action():
    action_space = ActionSpace()

    with pytest.raises(TypeError):
        action_space.validate("1")


def test_empty_action_space_is_rejected():
    with pytest.raises(ValueError):
        ActionSpace(actions=())


def test_duplicate_actions_are_rejected():
    with pytest.raises(ValueError):
        ActionSpace(actions=(0, 1, 1))


def test_non_integer_action_definition_is_rejected():
    with pytest.raises(TypeError):
        ActionSpace(actions=(0, "1", 2))