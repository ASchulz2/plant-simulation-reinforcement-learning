from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from agent.actor_critic import ActorCriticMLP


def save_policy(
    policy: ActorCriticMLP,
    path: str | Path,
) -> None:
    """Save actor-critic parameters."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        policy.state_dict(),
        path,
    )


def load_policy(
    policy: ActorCriticMLP,
    path: str | Path,
    *,
    device: torch.device | None = None,
) -> None:
    """Load actor-critic parameters."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Policy checkpoint not found: {path}"
        )

    state_dict: Any = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    policy.load_state_dict(
        state_dict
    )