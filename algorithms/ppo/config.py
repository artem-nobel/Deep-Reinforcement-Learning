from dataclasses import dataclass

import torch


@dataclass
class Config:

    env_name: str = "LunarLander-v3"
    # env_name: str = "FrozenLake-v1"
    # env_name: str = "CartPole-v1"
    # env_name: str = "Acrobot-v1"
    # env_name: str = "MountainCar-v0"
    # env_name: str = "CarRacing-v3"

    learning_rate: float = 1e-4
    # learning_rate: float = 3e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    rollout_size: int = 256

    clip_epsilon: float = 0.2
    ppo_epochs: int = 4


    n_episodes: int = 3000

    max_steps: int = 1000

    # seed: int = 5 #2700
    # seed: int = 24 #1050
    # seed: int = 128 #1200
    # seed: int = 224 #3000+
    seed: int = 310

    render: bool = False

    log_interval: int = 50

    critic_coef: float = 1

    num_envs: int = 4

    batch_size: int = 32

    device: torch.device = torch.device(
        "cpu"
    )