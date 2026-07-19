from dataclasses import dataclass

import torch


@dataclass
class Config:

    env_name: str = "LunarLander-v3"

    learning_rate: float = 1e-4

    gamma: float = 0.995

    n_episodes: int = 3000

    max_step: int = 1000

    seed: int = 5

    render: bool = False

    log_interval: int = 1

    # device: torch.device = torch.device(
    #     "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    device: torch.device = torch.device(
        "cpu"
    )