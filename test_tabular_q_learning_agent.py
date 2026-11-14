import pytest
import torch

from agent.tabular_q_learning_agent import (
    TabularQLearningAgent,
)
from environment.transition import Transition
from problem.observation import Observation
from problem.q_learning import (
    QLearningSettings,
    QLearningUpdater,
)
from problem.q_learning_agent import (
    QLearningAgent,
    QLearningAgentSettings,
)
from problem.q_table import QTable
from problem.problem import ManufacturingProblem
from problem.tabular_q_learning import TabularQLearner
from problem.tabular_state_encoder import (
    TabularStateEncoder,
)


def make_observation(
    *,
    simulation_time: float = 10.0,
    admission_buffer_contents=(
        "GER",
        "FR",
        "RUS",
    ),
    storage_a_count: int = 0,
    storage_a_language: str = "",
    storage_b_count: int = 0,
    storage_b_language: str = "",
    throughput: int = 0,
) -> Observation:
    return Observation(
        simulation_time=simulation_time,
        admission_buffer_contents=admission_buffer_contents,
        storage_a_count=storage_a_count,
        storage_a_language=storage_a_language,
        storage_b_count=storage_b_count,
        storage_b_language=storage_b_language,
        throughput=throughput,
    )


def make_agent(
    table: QTable | None = None,
) -> TabularQLearningAgent:
    if table is None:
        table = QTable()

    problem = ManufacturingProblem()

    q_agent = QLearningAgent(
        q_table=table,
        problem=problem,
        settings=QLearningAgentSettings(
            epsilon_start=1.0,
            epsilon_min=.05,
            epsilon_decay=.998
        ),
    )

    updater = QLearningUpdater(
        q_table=table,
        settings=QLearningSettings(
            learning_rate=0.5,
            discount_factor=0.9,
        ),
    )

    learner = TabularQLearner(
        agent=q_agent,
        updater=updater,
        problem=problem,
    )

    return TabularQLearningAgent(
        state_encoder=TabularStateEncoder(problem),
        learner=learner,
    )


def make_transition(
    *,
    reward: float,
    terminated: bool,
) -> Transition:
    zero = torch.tensor(0.0)

    return Transition(
        state=torch.zeros(1),
        action=0,
        reward=reward,
        next_state=torch.zeros(1),
        log_probability=zero,
        value=zero,
        terminated=terminated,
        progress_failure=False,
        succeeded=False,
        simulation_time=1.0,
        next_simulation_time=2.0,
        elapsed_time=1.0,
        throughput=0,
        next_throughput=0,
    )


def test_agent_selects_action_from_semantic_observation():
    agent = make_agent()

    observation = make_observation()

    action = agent.act(observation)

    assert action in (0, 1, 2)


def test_agent_selection_updates_state_visitation():
    agent = make_agent()

    observation = make_observation()

    agent.act(observation)
    agent.act(observation)

    state = agent.state_encoder.encode(
        observation
    )

    assert (
        agent.learner.state_registry.visits(state)
        == 2
    )


def test_agent_selection_updates_state_action_visitation():
    agent = make_agent()

    observation = make_observation()

    action = agent.act(observation)

    state = agent.state_encoder.encode(
        observation
    )

    assert (
        agent.learner.state_action_registry.visits(
            state,
            action,
        )
        == 1
    )


def test_agent_learning_updates_q_table():
    table = QTable()
    agent = make_agent(table)

    observation = make_observation(
        admission_buffer_contents=("GER",)
    )

    next_observation = make_observation(
        admission_buffer_contents=("FR",)
    )

    transition = make_transition(
        reward=10.0,
        terminated=True,
    )

    agent.learn(
        observation=observation,
        action=0,
        next_observation=next_observation,
        transition=transition,
    )

    state = agent.state_encoder.encode(
        observation
    )

    assert table.get(
        state,
        0,
    ) == pytest.approx(5.0)


def test_agent_learning_does_not_add_visitation():
    agent = make_agent()

    observation = make_observation(
        admission_buffer_contents=("GER",)
    )

    next_observation = make_observation(
        admission_buffer_contents=("FR",)
    )

    transition = make_transition(
        reward=1.0,
        terminated=True,
    )

    agent.learn(
        observation=observation,
        action=0,
        next_observation=next_observation,
        transition=transition,
    )

    state = agent.state_encoder.encode(
        observation
    )

    assert (
        agent.learner.state_registry.visits(state)
        == 0
    )

    assert (
        agent.learner.state_action_registry.visits(
            state,
            0,
        )
        == 0
    )


def test_agent_learning_distinguishes_storage_occupancy() -> None:
    table = QTable()
    agent = make_agent(table)

    observation_low = make_observation(
        admission_buffer_contents=("GER",),
        storage_a_language="GER",
        storage_a_count=2,
        storage_b_language="FR",
        storage_b_count=3,
    )

    observation_full = make_observation(
        admission_buffer_contents=("GER",),
        storage_a_language="GER",
        storage_a_count=10,
        storage_b_language="FR",
        storage_b_count=3,
    )

    transition = make_transition(
        reward=10.0,
        terminated=True,
    )

    agent.learn(
        observation=observation_low,
        action=0,
        next_observation=observation_full,
        transition=transition,
    )

    low_state = agent.state_encoder.encode(
        observation_low
    )

    full_state = agent.state_encoder.encode(
        observation_full
    )

    assert low_state != full_state

    assert table.get(
        low_state,
        0,
    ) == pytest.approx(5.0)

    assert table.get(
        full_state,
        0,
    ) == pytest.approx(0.0)