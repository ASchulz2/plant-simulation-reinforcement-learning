from __future__ import annotations

import torch

from agent.agent import (
    PolicyAgent,
    PolicyDecision,
)

from environment.transition import Transition

from learning.rollout_collector import (
    RolloutCollector,
    RolloutCollectorSettings,
)


class FakeAgent(PolicyAgent):

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


class FakeEnvironment:

    def __init__(self) -> None:
        self.actions: list[int] = []

    def step(
        self,
        *,
        action: int,
        log_probability: torch.Tensor,
        value: torch.Tensor,
    ) -> Transition:

        self.actions.append(action)

        return Transition(
            state=torch.tensor(
                [0.0],
                dtype=torch.float32,
            ),
            action=action,
            reward=1.0,
            next_state=torch.tensor(
                [1.0],
                dtype=torch.float32,
            ),
            log_probability=log_probability,
            value=value,
            terminated=False,
            progress_failure=False,
            succeeded=False,
            simulation_time=1.0,
            next_simulation_time=2.0,
            elapsed_time=1.0,
            throughput=0,
            next_throughput=1,
        )


def test_collector_uses_policy_agent_interface() -> None:

    environment = FakeEnvironment()
    agent = FakeAgent()

    collector = RolloutCollector(
        environment=environment,
        agent=agent,
        settings=RolloutCollectorSettings(
            horizon=3
        ),
    )

    observation = torch.tensor(
        [0.0],
        dtype=torch.float32,
    )

    (
        rollout,
        next_observation,
        terminated,
        progress_failure,
    ) = collector.collect(
        observation
    )

    assert rollout.length == 3

    assert environment.actions == [
        1,
        1,
        1,
    ]

    assert torch.equal(
        next_observation,
        torch.tensor(
            [1.0],
            dtype=torch.float32,
        ),
    )

    assert terminated is False
    assert progress_failure is False