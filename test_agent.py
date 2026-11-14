from __future__ import annotations

import pytest
import torch

from agent.agent import (
    Agent,
    AgentDecision,
    PolicyAgent,
    PolicyDecision,
)


class FakeAgent(Agent):

    def act(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> AgentDecision:

        return AgentDecision(
            action=1
        )


class FakePolicyAgent(PolicyAgent):

    def act(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicyDecision:

        return PolicyDecision(
            action=1,
            log_probability=torch.tensor(-0.5),
            value=torch.tensor(0.25),
        )


def test_agent_decision() -> None:
    decision = AgentDecision(
        action=1
    )

    assert decision.action == 1


def test_policy_decision() -> None:
    decision = PolicyDecision(
        action=1,
        log_probability=torch.tensor(-0.5),
        value=torch.tensor(0.25),
    )

    assert decision.action == 1

    assert torch.equal(
        decision.log_probability,
        torch.tensor(-0.5),
    )

    assert torch.equal(
        decision.value,
        torch.tensor(0.25),
    )


def test_generic_agent() -> None:
    agent = FakeAgent()

    observation = torch.zeros(
        4,
        dtype=torch.float32,
    )

    decision = agent.act(observation)

    assert isinstance(
        decision,
        AgentDecision,
    )

    assert decision.action == 1


def test_policy_agent() -> None:
    agent = FakePolicyAgent()

    observation = torch.zeros(
        4,
        dtype=torch.float32,
    )

    decision = agent.act(observation)

    assert isinstance(
        decision,
        PolicyDecision,
    )

    assert decision.action == 1


def test_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


def test_policy_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        PolicyAgent()  # type: ignore[abstract]