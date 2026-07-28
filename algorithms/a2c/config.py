from dataclasses import dataclass

import torch


@dataclass
class Config:

    # env_name: str = "LunarLander-v3"
    env_name: str = "CartPole-v1"

    learning_rate: float = 1e-4

    # gamma: float = 0.95
    # gamma: float = 0.85
    gamma: float = 0.8
    # gamma: float = 0.75

    n_episodes: int = 1500

    max_step: int = 500

    seed: int = 5

    render: bool = False

    log_interval: int = 1

    n_step: int = 5

    batch_size: int = 32

    # device: torch.device = torch.device(
    #     "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    device: torch.device = torch.device(
        "cpu"
    )