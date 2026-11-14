# Plant Simulation with Tabular Q-Learning

A reinforcement learning project that connects **Siemens Tecnomatix Plant Simulation 16.1** to a Python-based tabular Q-learning agent. The project explores how an agent can learn routing decisions in a manufacturing system through interaction with a discrete-event simulation.

The primary implementation is a tabular Q-learning baseline. Experimental actor-critic and PPO components are also present in the codebase but are not the focus of this project description.

## Project Overview

The simulated manufacturing system processes units (MUs) associated with different language categories. The agent makes routing decisions that determine whether an MU is assigned to one of two storage areas or returned.

The agent chooses between three actions:

| Action | Decision |
|---|---|
| 0 | Storage A |
| 1 | Storage B |
| 2 | Return |

The objective is to reach a target throughput while learning to make feasible routing decisions.

The implementation separates the reinforcement learning logic from the Plant Simulation interface. Python communicates with the simulation through a TCP socket using JSON messages, while COM automation is used for simulation lifecycle management and access to the simulation model.

## Architecture

The project is organized into separate components:

- **Simulation interface (`simulation/`)** — manages the Plant Simulation COM connection, model loading, and simulation lifecycle.
- **TCP communication (`tcp/`)** — implements the socket connection, message protocol, and decision/action handshake.
- **Environment (`environment/`)** — coordinates observations, transitions, rewards, episode execution, and termination.
- **Problem definition (`problem/`)** — defines the manufacturing problem, state representation, action space, and reward function.
- **Agent (`agent/`)** — contains the tabular Q-learning agent and experimental agent implementations.
- **Learning (`learning/`)** — contains learning and training utilities.
- **Visualization (`visualization/`)** — provides Q-table and experiment metric visualizations.

The TCP protocol coordinates the interaction between the simulation and the Python agent. Decision identifiers and acknowledgement identifiers are used to keep the exchange synchronized.

## Requirements and Setup

### Requirements

- Windows
- Python
- Siemens Tecnomatix Plant Simulation 16.1

Python dependencies are listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

The requirements file contains the following runtime and development dependencies:

```text
numpy==2.4.4
torch==2.13.0
matplotlib==3.11.1
pywin32==312
python-dotenv==1.2.4
pytest==9.1.1
```

### Plant Simulation model configuration

The Plant Simulation model is included in the repository:

```text
model/FirstGameBoardV0.0.1.spp
```

The model path is configured through an environment variable rather than being hard-coded in the source code.

After cloning the repository, create a local `.env` file from the supplied example:

```powershell
Copy-Item .env.example .env
```

Set `PLANT_SIMULATION_MODEL` in `.env` to the absolute path of the included model on your machine. For example:

```text
PLANT_SIMULATION_MODEL=D:/path/to/plant-simulation-reinforcement-learning/model/FirstGameBoardV0.0.1.spp
```

The `.env` file is intentionally excluded from version control. The `.env.example` file provides a template for local configuration.

The COM automation identifier and model root are configured in `config/settings.py`:

```python
prog_id = "Tecnomatix.PlantSimulation.RemoteControl.16.1"
model_path = ".Models.Model"
```

Additional settings control application visibility, model-loading delay, and COM diagnostics. TCP settings are configured separately in `config/tcp_settings.py`.

## Testing

The project uses `pytest` for regression testing.

Run the complete test suite from the project root:

```powershell
pytest -v
```

The current implementation passes **536 tests**.

A focused test module can be executed with:

```powershell
pytest test_q_learning_checkpoint.py -v
```

## Tabular Q-Learning

### Training

Training is split between two files:

- `run_tabular_q_learning.py` — executable entry point.
- `tabular_q_learning_main.py` — experiment configuration and orchestration.

Start a training run from the project root:

```powershell
python run_tabular_q_learning.py
```

The entry point configures the Plant Simulation interface and TCP connection before calling the experiment runner.

### Experiment configuration

The experiment settings are defined centrally in `tabular_q_learning_main.py`:

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

The agent uses an epsilon-greedy policy:

- Exploration starts at `epsilon = 1.0`.
- Exploration decays by a factor of `0.998` per action selection.
- The minimum exploration rate is `0.05`.

With `feasible=True`, actions that are invalid in the current state are excluded from action selection. The agent still learns preferences among the remaining feasible actions through its Q-values.

### State representation

The tabular agent uses an encoded representation of the manufacturing system state. It includes the current MU's language category, the language counts in the admission buffer, and the language and occupancy information of the two storage areas.

The resulting state representation is used as the key in a sparse Q-table. Only encountered state-action values need to be stored.

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

This configuration emphasizes feasible decisions and terminal success or failure rather than additional reward-shaping terms.

### Checkpoints

The training checkpoint is stored at:

```text
checkpoint/tabular_q_learning_pure.pkl
```

The checkpoint is generated during training and is not included in the repository. Run the training script first to create it before starting policy evaluation.

A checkpoint is saved after each completed episode. It contains the sparse Q-table, agent state, episode and decision counters, random-number-generator states, and experiment metadata.

The metadata records relevant experiment settings, including the state representation, action space, learning configuration, reward configuration, feasibility setting, and episode horizon.

Training can be resumed by enabling `resume=True` in the call to `run()` in `run_tabular_q_learning.py`.

### Training metrics

The training run reports episode outcomes and metrics such as:

- Decision points
- Mean reward and throughput
- Drain-event statistics
- Action usage
- Distinct visited states
- Distinct visited state-action pairs

It also generates visualizations of total reward, throughput, decision points, mean drain count, and simulation time.

## Policy Evaluation

Evaluation is implemented separately from training:

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

This checkpoint must first be generated by running the training script. It is not included in the repository.

The number of evaluation episodes is configurable through `num_episodes`.

### Evaluation policy

During evaluation, epsilon-greedy exploration is disabled:

```text
epsilon = 0
```

The agent therefore follows a greedy policy based on the learned Q-table. No Q-value updates are performed during evaluation.

Feasibility can be configured independently. The current evaluation configuration uses:

```python
EVALUATION_SETTINGS = EvaluationSettings(
    feasible=True,
)
```

### Q-table visualization

The evaluation produces two Q-table visualizations:

1. **Top learned Q-values** — displays state-action pairs with the largest absolute Q-values.
2. **Action values for an encountered state** — compares the stored values of the available actions for a state observed during evaluation.

These visualizations provide insight into the learned action preferences. Since the Q-table is sparse, unseen state-action pairs retain the table's default value.

### Evaluation metrics

The evaluation reports checkpoint information, policy configuration, episode outcomes, simulation time, reward and throughput metrics, decision points, drain-event statistics, action usage, and the number of distinct visited states and state-action pairs.

Evaluation therefore measures the restored policy independently of further learning.