from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from agent.agent import PolicyAgent
from environment.plant_environment import PlantEnvironment
from environment.rollout import Rollout


@dataclass(frozen=True)
class RolloutCollectorSettings:
    """Configuration of rollout collection."""

    horizon: int = 1024


class RolloutCollector:
    """
    Collect one bounded rollout from a PlantEnvironment.

    The collector requires a policy-capable agent because PPO
    rollouts store action log-probabilities and state values.
    """

    def __init__(
        self,
        environment: PlantEnvironment,
        agent: PolicyAgent,
        settings: RolloutCollectorSettings | None = None,
    ) -> None:

        self.environment = environment
        self.agent = agent

        self.settings = (
            settings
            if settings is not None
            else RolloutCollectorSettings()
        )

        if self.settings.horizon <= 0:
            raise ValueError(
                "Rollout horizon must be positive."
            )

    def collect(
        self,
        observation: Tensor,
    ) -> tuple[Rollout, Tensor, bool, bool]:
        """
        Collect one rollout from the current episode.
        """

        rollout = Rollout(
            horizon=self.settings.horizon
        )

        terminated = False
        progress_failure = False

        while not rollout.is_full:

            # --------------------------------------------------
            # Sample action from the policy agent.
            # --------------------------------------------------

            with torch.no_grad():

                decision = self.agent.act(
                    observation
                )

            # --------------------------------------------------
            # Execute one environment transition.
            # --------------------------------------------------

            transition = self.environment.step(
                action=decision.action,
                log_probability=(
                    decision.log_probability
                ),
                value=decision.value,
            )

            # --------------------------------------------------
            # Store transition.
            # --------------------------------------------------

            rollout.add(
                transition
            )

            # --------------------------------------------------
            # Continue from the resulting observation.
            # --------------------------------------------------

            observation = transition.next_state

            # --------------------------------------------------
            # Stop early if the episode terminated.
            # --------------------------------------------------

            if transition.terminated:

                terminated = True

                progress_failure = (
                    transition.progress_failure
                )

                break

        return (
            rollout,
            observation,
            terminated,
            progress_failure,
        )