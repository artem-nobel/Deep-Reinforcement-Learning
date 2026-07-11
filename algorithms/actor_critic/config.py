from dataclasses import dataclass

import torch


@dataclass
class Config:

    env_name: str = "LunarLander-v3"

    learning_rate: float = 3e-4

    gamma: float = 0.99

    n_episodes: int = 100

    max_step: int = 1000

    seed: int = 5

    render: bool = False

    log_interval: int = 10

    device: torch.device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )