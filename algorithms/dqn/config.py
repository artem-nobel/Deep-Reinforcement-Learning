from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Environment
    env_name: str = "LunarLander-v3"

    # Learning
    learning_rate: float = 1e-4
    gamma: float = 0.985

    # Exploration
    epsilon_start: float = 0.5
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.997

    # Replay Buffer
    buffer_size: int = 100000
    batch_size: int = 128

    # Training
    n_episodes: int = 500
    max_step: int = 1000

    # DQN
    target_update_freq: int = 5
    train_freq: int = 1

    # Vectorized Environments
    n_envs: int = 8

    # Prioritized Experience Replay
    alpha: float = 0.8
    beta_start: float = 0.4
    beta_increment: float = 0.01

    # Misc
    visualize_every: int = 999
    seed: int = 5

    device: torch.device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )