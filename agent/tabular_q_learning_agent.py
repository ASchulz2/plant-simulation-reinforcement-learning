from __future__ import annotations

from dataclasses import dataclass

from problem.observation import Observation
from problem.tabular_q_learning import TabularQLearner
from problem.tabular_state_encoder import TabularStateEncoder
from problem.tabular_transition import TabularTransition
from environment.transition import Transition


@dataclass
class TabularQLearningAgent:
    """
    Adapter between semantic environment observations and
    the tabular Q-learning machinery.

    The agent does not know about Plant Simulation, TCP,
    or tensors.
    """

    state_encoder: TabularStateEncoder
    learner: TabularQLearner

    def act(
        self,
        observation: Observation,
    ) -> int:
        """
        Encode a semantic observation and select an action.
        """

        state = self.state_encoder.encode(
            observation
        )

        return self.learner.select_action(
            state
        )

    def learn(
        self,
        *,
        observation: Observation,
        action: int,
        next_observation: Observation,
        transition: Transition,
    ) -> float:
        """
        Learn from one environment transition.
        """

        state = self.state_encoder.encode(
            observation
        )

        next_state = self.state_encoder.encode(
            next_observation
        )

        tabular_transition = TabularTransition(
            state=state,
            action=action,
            reward=float(transition.reward),
            next_state=next_state,
            terminated=transition.terminated,
        )

        return self.learner.learn(
            tabular_transition
        )