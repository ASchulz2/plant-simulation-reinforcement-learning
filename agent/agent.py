from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class AgentDecision:
    """Result of a generic agent decision."""

    action: int


@dataclass(frozen=True)
class PolicyDecision(AgentDecision):
    """
    Result of a stochastic policy decision.

    These additional quantities are required by PPO-style
    rollout collection.
    """

    log_probability: Tensor
    value: Tensor


class Agent(ABC):
    """
    Generic decision-making interface.

    A concrete agent receives an observation and returns an action.
    """

    @abstractmethod
    def act(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
    ) -> AgentDecision:
        """Select an action for the observation."""
        raise NotImplementedError


class PolicyAgent(Agent, ABC):
    """
    Agent interface for stochastic policy-based learning.

    In addition to the selected action, a policy agent returns
    the action log-probability and a state-value estimate.
    """

    @abstractmethod
    def act(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicyDecision:
        """Select an action and return policy metadata."""
        raise NotImplementedError