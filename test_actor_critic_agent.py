from __future__ import annotations

import torch

from agent.actor_critic import (
    ActorCriticMLP,
    ActorCriticSettings,
)
from agent.actor_critic_agent import (
    ActorCriticAgent,
)
from agent.agent import PolicyDecision


def make_agent() -> ActorCriticAgent:
    settings = ActorCriticSettings(
        observation_dim=4,
        action_dim=3,
        hidden_dim=16,
        hidden_layers=2,
        activation="tanh",
    )

    policy = ActorCriticMLP(settings)

    return ActorCriticAgent(policy)


def test_agent_is_constructed() -> None:
    agent = make_agent()

    assert isinstance(
        agent,
        ActorCriticAgent,
    )


def test_stochastic_action_output() -> None:
    agent = make_agent()

    observation = torch.zeros(
        4,
        dtype=torch.float32,
    )

    output = agent.act(observation)

    assert isinstance(
        output,
        PolicyDecision,
    )

    assert isinstance(
        output.action,
        int,
    )

    assert 0 <= output.action < 3

    assert output.log_probability.ndim == 0
    assert output.value.ndim == 0


def test_deterministic_action_output() -> None:
    agent = make_agent()

    observation = torch.zeros(
        4,
        dtype=torch.float32,
    )

    first = agent.act(
        observation,
        deterministic=True,
    )

    second = agent.act(
        observation,
        deterministic=True,
    )

    assert first.action == second.action

    assert torch.equal(
        first.log_probability,
        second.log_probability,
    )

    assert torch.equal(
        first.value,
        second.value,
    )


def test_stochastic_actions_are_valid() -> None:
    agent = make_agent()

    observation = torch.zeros(
        4,
        dtype=torch.float32,
    )

    actions = {
        agent.act(observation).action
        for _ in range(50)
    }

    assert actions.issubset({0, 1, 2})