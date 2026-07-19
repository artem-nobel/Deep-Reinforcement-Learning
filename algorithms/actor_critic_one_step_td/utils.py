import random

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


def get_env_info(env):

    if not isinstance(
            env.action_space,
            gym.spaces.Discrete
    ):
        raise ValueError(
            "Only discrete action space supported."
        )

    n_states = env.observation_space.shape[0]

    n_actions = env.action_space.n

    return n_states, n_actions


def to_tensor(state, device):

    return torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    )