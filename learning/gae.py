from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GAESettings:
    """Configuration for Generalized Advantage Estimation."""

    gamma: float = 0.99
    gae_lambda: float = 0.95


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    settings: GAESettings,
) -> tuple[Tensor, Tensor]:
    """
    Compute GAE advantages and value targets.

    Returns
    -------
    advantages:
        Estimated advantage A_t for every transition.

    returns:
        Value targets R_t = A_t + V(s_t).
    """

    if rewards.ndim != 1:
        raise ValueError(
            "rewards must have shape [T]."
        )

    if values.ndim != 1:
        raise ValueError(
            "values must have shape [T]."
        )

    if next_values.ndim != 1:
        raise ValueError(
            "next_values must have shape [T]."
        )

    if terminated.ndim != 1:
        raise ValueError(
            "terminated must have shape [T]."
        )

    horizon = rewards.shape[0]

    if not (
        values.shape[0]
        == next_values.shape[0]
        == terminated.shape[0]
        == horizon
    ):
        raise ValueError(
            "GAE tensors must have identical "
            "first dimensions."
        )

    advantages = torch.zeros_like(
        rewards
    )

    gae = torch.zeros(
        (),
        dtype=rewards.dtype,
        device=rewards.device,
    )

    for t in reversed(range(horizon)):

        not_terminal = 1.0 - terminated[t]

        delta = (
            rewards[t]
            + settings.gamma
            * not_terminal
            * next_values[t]
            - values[t]
        )

        gae = (
            delta
            + settings.gamma
            * settings.gae_lambda
            * not_terminal
            * gae
        )

        advantages[t] = gae

    returns = advantages + values

    return advantages, returns