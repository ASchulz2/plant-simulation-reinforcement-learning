from __future__ import annotations

import random

from problem.problem import ManufacturingProblem
from problem.q_learning import (
    QLearningSettings,
    QLearningUpdater,
)
from problem.q_learning_agent import (
    QLearningAgent,
    QLearningAgentSettings,
)
from problem.q_table import QTable
from problem.tabular_q_learning import TabularQLearner
from problem.tabular_state import TabularState
from problem.tabular_transition import TabularTransition


def make_state(
    current_language: str,
    storage_a_language: str = "",
    storage_a_count: int = 0,
    storage_b_language: str = "",
    storage_b_count: int = 0,
) -> TabularState:
    return TabularState(
        current_language=current_language,
        admission_buffer_ger_count=(
            1
            if current_language == "GER"
            else 0
        ),
        admission_buffer_fr_count=(
            1
            if current_language == "FR"
            else 0
        ),
        admission_buffer_rus_count=(
            1
            if current_language == "RUS"
            else 0
        ),
        storage_a_language=storage_a_language,
        storage_a_count=storage_a_count,
        storage_b_language=storage_b_language,
        storage_b_count=storage_b_count,
    )


def make_learner(
    table: QTable,
) -> TabularQLearner:
    problem = ManufacturingProblem()

    agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=1.0,
            epsilon_min=.05,
            epsilon_decay=.998
        ),
        rng=random.Random(123),
    )

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(
            learning_rate=0.5,
            discount_factor=0.9,
        ),
    )

    return TabularQLearner(
        agent=agent,
        updater=updater,
        problem=problem,
    )


def test_q_learning_propagates_terminal_value_backward():
    table = QTable()
    learner = make_learner(table)

    s0 = make_state("GER")
    s1 = make_state("FR")
    s2 = make_state("RUS")

    # The deterministic optimal path is:
    #
    # s0 --1--> s1 --2--> s2 --0--> terminal
    #
    # Only the final transition produces reward.

    episodes = 200

    for _ in range(episodes):
        transition_0 = TabularTransition(
            state=s0,
            action=1,
            reward=0.0,
            next_state=s1,
            terminated=False,
        )

        learner.learn(transition_0)

        transition_1 = TabularTransition(
            state=s1,
            action=2,
            reward=0.0,
            next_state=s2,
            terminated=False,
        )

        learner.learn(transition_1)

        transition_2 = TabularTransition(
            state=s2,
            action=0,
            reward=10.0,
            next_state=s2,
            terminated=True,
        )

        learner.learn(transition_2)

    assert table.get(s2, 0) > 0.0
    assert table.get(s1, 2) > 0.0
    assert table.get(s0, 1) > 0.0


def test_q_learning_prefers_learned_actions():
    table = QTable()
    learner = make_learner(table)

    s0 = make_state("GER")
    s1 = make_state("FR")
    s2 = make_state("RUS")

    for _ in range(200):
        learner.learn(
            TabularTransition(
                state=s0,
                action=1,
                reward=0.0,
                next_state=s1,
                terminated=False,
            )
        )

        learner.learn(
            TabularTransition(
                state=s1,
                action=2,
                reward=0.0,
                next_state=s2,
                terminated=False,
            )
        )

        learner.learn(
            TabularTransition(
                state=s2,
                action=0,
                reward=10.0,
                next_state=s2,
                terminated=True,
            )
        )

    # Exploitation behavior is tested independently by using epsilon=0.
    greedy_agent = QLearningAgent(
        q_table=table,
        problem=ManufacturingProblem(),
        settings=QLearningAgentSettings(
            epsilon_start=0.,
            epsilon_min=.0,
            epsilon_decay=.998
        ),
    )

    assert greedy_agent.select_action(s0) == 1
    assert greedy_agent.select_action(s1) == 2
    assert greedy_agent.select_action(s2) == 0
