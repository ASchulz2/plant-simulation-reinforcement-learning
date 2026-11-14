from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ActorCriticSettings:
    """Configuration of the actor-critic MLP."""

    observation_dim: int
    action_dim: int = 3

    hidden_dim: int = 128
    hidden_layers: int = 2

    activation: str = "tanh"


class ActorCriticMLP(nn.Module):
    """
    Shared MLP with separate actor and critic heads.

    Input:
        float32 observation tensor

    Outputs:
        actor logits
        scalar state-value estimate
    """

    def __init__(
        self,
        settings: ActorCriticSettings,
    ) -> None:

        super().__init__()

        self.settings = settings

        self.shared = self._build_shared_network()

        self.actor = nn.Linear(
            settings.hidden_dim,
            settings.action_dim,
        )

        self.critic = nn.Linear(
            settings.hidden_dim,
            1,
        )

        self._initialize_weights()

    def _build_shared_network(
        self,
    ) -> nn.Sequential:

        layers: list[nn.Module] = []

        input_dim = self.settings.observation_dim

        for _ in range(
            self.settings.hidden_layers
        ):

            layers.append(
                nn.Linear(
                    input_dim,
                    self.settings.hidden_dim,
                )
            )

            layers.append(
                self._activation()
            )

            input_dim = self.settings.hidden_dim

        return nn.Sequential(*layers)

    def _activation(self) -> nn.Module:

        if self.settings.activation == "tanh":
            return nn.Tanh()

        if self.settings.activation == "relu":
            return nn.ReLU()

        raise ValueError(
            f"Unsupported activation: "
            f"{self.settings.activation!r}"
        )

    def _initialize_weights(self) -> None:
        """
        Initialize the actor-critic network.

        Shared layers:
            Orthogonal initialization with gain sqrt(2).

        Actor:
            Small orthogonal initialization to produce an
            approximately uniform initial policy.

        Critic:
            Orthogonal initialization with unit gain.

        All biases:
            Zero.
        """

        # ------------------------------------------------------
        # Shared network
        # ------------------------------------------------------

        for module in self.shared:

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.orthogonal_(
                    module.weight,
                    gain=nn.init.calculate_gain(
                        self.settings.activation
                    ),
                )

                nn.init.zeros_(
                    module.bias
                )

        # ------------------------------------------------------
        # Actor head
        # ------------------------------------------------------

        nn.init.orthogonal_(
            self.actor.weight,
            gain=0.01,
        )

        nn.init.zeros_(
            self.actor.bias
        )

        # ------------------------------------------------------
        # Critic head
        # ------------------------------------------------------

        nn.init.orthogonal_(
            self.critic.weight,
            gain=1.0,
        )

        nn.init.zeros_(
            self.critic.bias
        )

    def forward(
        self,
        observation: Tensor,
    ) -> tuple[Tensor, Tensor]:

        hidden = self.shared(
            observation
        )

        logits = self.actor(
            hidden
        )

        value = self.critic(
            hidden
        ).squeeze(-1)

        return logits, value

    def sample_action(
        self,
        observation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        logits, value = self.forward(
            observation
        )

        distribution = torch.distributions.Categorical(
            logits=logits
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return (
            action,
            log_probability,
            value,
        )

    def deterministic_action(
        self,
        observation: Tensor,
    ) -> Tensor:

        logits, _ = self.forward(
            observation
        )

        return torch.argmax(
            logits,
            dim=-1,
        )