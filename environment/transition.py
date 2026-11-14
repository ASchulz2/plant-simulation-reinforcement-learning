from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from environment.emptying_event import EmptyingEvent
from environment.termination import TerminationReason


@dataclass(frozen=True)
class Transition:
    """
    One reinforcement-learning transition.

    A transition corresponds to exactly one robot decision
    interval in Plant Simulation.
    """

    state: Tensor
    action: int
    reward: float
    next_state: Tensor

    log_probability: Tensor
    value: Tensor

    terminated: bool
    progress_failure: bool
    succeeded: bool

    elapsed_time: float

    simulation_time: float
    next_simulation_time: float

    throughput: int
    next_throughput: int

    emptying_events: tuple[EmptyingEvent, ...] = ()

    termination_reason: TerminationReason | None = None