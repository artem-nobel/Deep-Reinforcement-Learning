import numpy as np
import torch
from matplotlib import pyplot as plt



def get_env_info(env):
    if hasattr(env.observation_space, "shape"):
        n_states = env.observation_space.shape[0]
    else:
        raise ValueError("Unsupported observation space")

    if hasattr(env.action_space, "n"):
        n_actions = env.action_space.n
    else:
        raise ValueError("Only discrete actions supported")

    return n_states, n_actions

def _to_tensor(state):
    return torch.tensor(state, dtype=torch.float32)

def print_plot(config, reward_history, mean_reward_history):
    print('max reward', max(reward_history))
    print('mean reward', np.mean(reward_history))

    print(f"Mean last 100: {np.mean(reward_history[-100:]):.1f}")

    plt.plot(reward_history, label="reward_history", color="red")
    plt.plot(mean_reward_history, label="mean_reward_history", color="blue")
    plt.title("dqn_duel_per")
    plt.text(0., 500, f'LR = {config.learning_rate:.4f}, gamma = {config.gamma:.3f}')
    plt.text(0., 450, f'PER : ALPHA = {config.alpha:.2f}')

    plt.text(0., 400, f'Seed = {config.seed:.2f}, target_up = {config.target_update_freq:.2f}')
    plt.text(0., 350,f"Mean last 100: {np.mean(reward_history[-100:]):.1f}")


    plt.ylim(-400, 600)
    plt.legend()
    plt.grid(True)
    plt.show()