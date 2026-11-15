# Plant Simulation with Tabular Q-Learning

A reinforcement learning project integrating **Siemens Tecnomatix Plant Simulation 16.1** with a Python-based tabular Q-learning agent. The project investigates how an agent can learn routing decisions in a manufacturing system through interaction with a discrete-event simulation.

The main implementation is a tabular Q-learning baseline. Experimental actor-critic and Proximal Policy Optimization (PPO) components are also present in the codebase but are outside the scope of the current project description.

## Project Overview

The simulated manufacturing system processes material units (MUs) belonging to different language categories. At designated decision points, the agent determines how the current MU should be routed.

The action space consists of three discrete actions:

| Action | Routing decision |
|:---:|---|
| 0 | Assign to Storage A |
| 1 | Assign to Storage B |
| 2 | Return |

The objective is to achieve a specified production throughput while selecting feasible routing decisions.

The project connects a discrete-event simulation to a reinforcement learning workflow. The Python implementation manages the agent, state representation, reward calculation, and learning process, while Plant Simulation provides the manufacturing-system dynamics.

A deliberately separated interface architecture keeps the reinforcement learning logic independent of the simulation's COM automation layer.

## Architecture

The implementation is organized into components with distinct responsibilities:

| Component | Responsibility |
|---|---|
| `simulation/` | Plant Simulation COM connection, model loading, and simulation lifecycle |
| `tcp/` | TCP socket communication, JSON message protocol, and decision/action handshake |
| `environment/` | Environment interaction, observations, transitions, episode execution, and termination |
| `problem/` | Manufacturing problem definition, state representation, action space, and reward function |
| `agent/` | Tabular Q-learning agent and experimental agent implementations |
| `learning/` | Learning algorithms and training utilities |
| `visualization/` | Q-table diagnostics and experiment metric visualizations |
| `config/` | Simulation and communication settings |

### Simulation–Python communication

Python communicates with Plant Simulation through a TCP socket using JSON messages.

The communication protocol coordinates the exchange of observations and routing actions. Decision-point identifiers and action acknowledgement identifiers are used to synchronize the two systems and ensure that actions are associated with the corresponding simulation decisions.

COM automation is kept separate from this interaction protocol. It is used for simulation lifecycle management and access to the simulation model, while the TCP interface handles the decision/action exchange.

This separation allows the reinforcement learning components to operate independently of the details of the COM interface.

## Requirements and Setup

### Requirements

- Windows
- Python
- Siemens Tecnomatix Plant Simulation 16.1

The Python dependencies are specified in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

The current requirements include:

```text
numpy==2.4.4
torch==2.13.0
matplotlib==3.11.1
pywin32==312
python-dotenv==1.2.4
pytest==9.1.1
```

PyTorch is included for the experimental neural-network-based components. The primary tabular Q-learning implementation uses a sparse Q-table.

### Plant Simulation model configuration

The Plant Simulation model is included in the repository:

```text
model/FirstGameBoardV0.0.1.spp
```

The model path is configured through an environment variable rather than being hard-coded in the application.

After cloning the repository, create a local `.env` file from the supplied example:

```powershell
Copy-Item .env.example .env
```

Set `PLANT_SIMULATION_MODEL` in `.env` to the absolute path of the model on your machine. For example:

```text
PLANT_SIMULATION_MODEL=D:/path/to/plant-simulation-reinforcement-learning/model/FirstGameBoardV0.0.1.spp
```

The `.env` file is intentionally excluded from version control. The `.env.example` file provides a template for local configuration.

The COM automation identifier and model root are configured in `config/settings.py`:

```python
prog_id = "Tecnomatix.PlantSimulation.RemoteControl.16.1"
model_path = ".Models.Model"
```

Additional settings control application visibility, model-loading delay, and COM diagnostics. TCP connection settings are maintained separately in `config/tcp_settings.py`.

## Testing

The project uses `pytest` for regression testing.

Run the complete test suite from the project root:

```powershell
pytest -v
```

The current implementation passes **536 tests**.

The test suite covers, among other things:

- Simulation interface and environment behavior
- TCP message protocol and communication
- Decision/action synchronization
- State representation and reward calculation
- Q-learning agent behavior
- Episode execution and termination
- Checkpoint persistence and restoration
- Training and evaluation workflows

A focused test module can be executed independently:

```powershell
pytest test_q_learning_checkpoint.py -v
```

## Tabular Q-Learning

### Training workflow

Training is implemented in two main files:

- `run_tabular_q_learning.py` — executable entry point.
- `tabular_q_learning_main.py` — experiment configuration and orchestration.

Start a training run from the project root:

```powershell
python run_tabular_q_learning.py
```

The entry point configures the Plant Simulation interface and TCP connection before starting the experiment runner.

The experiment runner coordinates episodes, interacts with the environment, updates the Q-table, collects metrics, and manages checkpoints.

### Experiment configuration

The experiment settings are defined centrally in `tabular_q_learning_main.py`.

The current configuration is:

```python
EXPERIMENT_SETTINGS = QLearningExperimentSettings(
    target_throughput=1000,
    storage_capacity=10,
    admission_buffer_capacity=4096,
    max_steps=4096,
    actions=(0, 1, 2),
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.998,
    reward_settings=REWARD_SETTINGS,
    feasible=True,
)
```

The agent uses an epsilon-greedy action-selection policy:

- Exploration starts at `epsilon = 1.0`.
- The exploration rate decays by a factor of `0.998` per action selection.
- The minimum exploration rate is `0.05`.

The learning parameters are:

| Parameter | Value |
|---|---:|
| Learning rate | 0.1 |
| Discount factor | 0.99 |
| Initial exploration rate | 1.0 |
| Minimum exploration rate | 0.05 |
| Exploration decay | 0.998 |

### Feasible action selection

The experiment uses `feasible=True` to restrict action selection to actions that are valid in the current state.

An action mask excludes infeasible actions from the available action set. The agent therefore learns Q-values and selects actions among those permitted by the current manufacturing-system state.

The feasibility mechanism defines the admissible actions; the Q-table determines the agent's learned preferences among them.

### State representation

The tabular agent uses an encoded representation of the manufacturing-system state.

The state includes:

- The language category of the current MU
- The language counts in the admission buffer
- The language and occupancy information of Storage A
- The language and occupancy information of Storage B

The encoded state serves as the key in a sparse Q-table. Only encountered state-action values need to be stored, rather than allocating a dense table for the entire theoretical state space.

### Reward configuration

The current experiment uses the following reward settings:

```python
REWARD_SETTINGS = QLearningRewardSettings(
    throughput_weight=0.0,
    storage_reward=0.0,
    time_penalty=0.0,
    return_waiting_time_penalty=0.0,
    return_waiting_time_reference=21.0,
    success_bonus=10.0,
    failure_penalty=10.0,
    premature_emptying_penalty=0.0,
    return_action_penalty=0.0,
    feasibility_reward=1.0,
)
```

The reward configuration emphasizes feasible decisions and successful episode completion:

| Reward component | Setting |
|---|---:|
| Feasible decision reward | +1.0 |
| Successful termination bonus | +10.0 |
| Manufacturing failure penalty | 10.0 |
| Throughput reward shaping | Disabled |
| Storage reward shaping | Disabled |
| Time penalty | Disabled |
| Return waiting-time penalty | Disabled |
| Premature emptying penalty | Disabled |
| Return action penalty | Disabled |

Consequently, the current experiment rewards feasible routing decisions and reaching the target throughput, without additional shaping terms for storage utilization, waiting time, or routing preferences.

### Checkpoints and resuming training

The training checkpoint is stored at:

```text
checkpoint/tabular_q_learning_pure.pkl
```

The checkpoint is generated during training and is not included in the repository. Run the training script first to create it before starting policy evaluation.

A checkpoint is saved after each completed episode. It contains the sparse Q-table, agent state, episode and decision counters, random-number-generator states, and experiment metadata.

The metadata records relevant experiment settings, including the state representation, action space, learning configuration, reward configuration, feasibility setting, and episode horizon.

Training can be resumed by enabling `resume=True` in the call to `run()` in `run_tabular_q_learning.py`.

### Training metrics

The training workflow collects episode-level metrics, including:

- Episode outcome
- Decision points
- Total reward
- Throughput
- Simulation time
- Emptying-event statistics
- Action usage
- Distinct visited states
- Distinct visited state-action pairs

The experiment also generates visualizations of:

- Total reward per episode
- Throughput per episode
- Decision points per episode
- Mean MUs per emptying event
- Simulation time per episode

These metrics provide information about learning progress, production performance, routing behavior, and the computational effort required to complete an episode.

## Policy Evaluation

Policy evaluation is implemented separately from training:

- `tabular_q_learning_evaluate_main.py` — executable entry point.
- `tabular_q_learning_evaluate.py` — checkpoint restoration and evaluation orchestration.

Run the evaluation with:

```powershell
python tabular_q_learning_evaluate_main.py
```

The evaluator restores the checkpoint from:

```text
checkpoint/tabular_q_learning_pure.pkl
```

The checkpoint must first be generated by running the training script. It is not included in the repository.

The number of evaluation episodes is configurable through `num_episodes`.

### Evaluation policy

During evaluation, epsilon-greedy exploration is disabled:

```text
epsilon = 0
```

The agent therefore follows a greedy policy based on the learned Q-table. No Q-value updates are performed during evaluation.

Feasibility is configured independently of the training workflow. The current evaluation configuration uses:

```python
EVALUATION_SETTINGS = EvaluationSettings(
    feasible=True,
)
```

This evaluates the learned policy while retaining the same feasibility constraint used during training.

### Q-table visualization

The evaluation produces two Q-table visualizations:

1. **Top learned Q-values** — displays state-action pairs with the largest absolute Q-values.
2. **Action values for an encountered state** — compares the stored values of available actions for a state observed during evaluation.

These visualizations provide diagnostic insight into the learned action preferences.

Because the Q-table is sparse, unseen state-action pairs retain the table's default value. Consequently, the visualizations should be interpreted in the context of the states encountered during training and evaluation.

### Evaluation metrics

The evaluation reports:

- Checkpoint information
- Policy configuration
- Episode outcomes
- Simulation time
- Total reward
- Throughput
- Decision points
- Emptying-event statistics
- Action usage
- Distinct visited states
- Distinct visited state-action pairs

Evaluation measures the restored policy independently of further learning. This makes it possible to assess the behavior of the learned greedy policy separately from the exploratory training process.

## Project Scope

The current project focuses on the implementation and evaluation of a tabular Q-learning baseline for routing decisions in a discrete-event manufacturing simulation.

The main engineering aspects are:

- Integration of a commercial discrete-event simulation with a Python reinforcement learning environment
- Separation of simulation lifecycle management and decision/action communication
- Explicit synchronization of simulation decisions and agent actions
- State and reward design for a manufacturing routing problem
- Sparse tabular Q-learning with feasible action selection
- Checkpoint-based training and independent greedy policy evaluation

Experimental actor-critic and PPO components are also present in the repository, but the results and discussion in this README focus on the tabular Q-learning implementation.

## Experimental Results

### Training

The tabular Q-learning agent was trained for 10 episodes using the experiment configuration described above.

The training run completed all 10 episodes and achieved a mean throughput of 1,000 MUs per episode, corresponding to the configured target throughput.

| Metric | Result |
|---|---:|
| Training episodes | 10 |
| Mean episode reward | 2,807.5 |
| Mean throughput | 1,000 |
| Total decision points | 27,975 |
| Total emptying events | 2,077 |
| Total MUs processed through emptying events | 14,164 |
| Mean MUs per emptying event | 6.82 |
| Distinct visited states | 25,203 |
| Distinct visited state-action pairs | 25,224 |

The observed action distribution was:

| Action | Selections | Relative frequency |
|---|---:|---:|
| Storage A | 9,256 | 33.09% |
| Storage B | 4,955 | 17.71% |
| Return | 13,764 | 49.20% |
| **Total** | **27,975** | **100.00%** |

The training results show that the agent reached the target throughput while exploring a substantial number of distinct states and state-action pairs.

### Greedy policy evaluation

After training, the saved checkpoint was restored and evaluated using a greedy policy with exploration disabled (`epsilon = 0`).

The evaluation completed one episode successfully and reached the target throughput.

| Metric | Result |
|---|---:|
| Evaluation episodes | 1 |
| Episode outcome | Success |
| Throughput | 1,000 |
| Total reward | 3,195 |
| Decision points | 3,185 |
| Simulation time | 38,683.78 |
| Emptying events | 256 |
| MUs processed through emptying events | 1,595 |
| Mean MUs per emptying event | 6.23 |
| Distinct visited states | 2,899 |
| Distinct visited state-action pairs | 2,899 |

The greedy policy's action distribution was:

| Action | Selections | Relative frequency |
|---|---:|---:|
| Storage A | 1,008 | 31.65% |
| Storage B | 589 | 18.49% |
| Return | 1,588 | 49.86% |
| **Total** | **3,185** | **100.00%** |

### Discussion

The greedy evaluation achieved the target throughput without further Q-value updates. Its action distribution was similar to the aggregate distribution observed during training, with the Return action remaining the most frequently selected action.

The evaluation also visited 2,899 distinct states and state-action pairs, illustrating the number of different decisions encountered during a single episode.

The training and evaluation rewards should not be compared directly as measures of policy improvement: they correspond to episodes with different numbers of decision points. In particular, the evaluation episode involved more decisions than the average training episode.

The evaluation currently covers one episode and should therefore be interpreted as an initial assessment of the learned policy rather than a statistically robust estimate of its performance.

### Result visualizations

The training and evaluation workflows generate plots to inspect learning behavior, throughput, decision counts, emptying-event statistics, simulation time, and learned Q-values.

These visualizations complement the aggregate metrics by showing episode-level variation and the distribution of learned action values.