from __future__ import annotations

import torch
from torch import Tensor

from agent.actor_critic import ActorCriticMLP
from agent.agent import PolicyAgent, PolicyDecision


class ActorCriticAgent(PolicyAgent):
    """Agent backed by the standard actor-critic MLP."""

    def __init__(
        self,
        policy: ActorCriticMLP,
    ) -> None:
        self.policy = policy

    def act(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicyDecision:

        if deterministic:
            action = self.policy.deterministic_action(
                observation
            )

            logits, value = self.policy.forward(
                observation
            )

            distribution = torch.distributions.Categorical(
                logits=logits
            )

            log_probability = distribution.log_prob(
                action
            )

        else:
            (
                action,
                log_probability,
                value,
            ) = self.policy.sample_action(
                observation
            )

        return PolicyDecision(
            action=int(action.item()),
            log_probability=log_probability,
            value=value,
        )