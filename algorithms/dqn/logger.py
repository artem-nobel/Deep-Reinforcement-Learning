import mlflow
from matplotlib import pyplot as plt
import numpy as np


# def print_plot(reward_history, mean_reward_history):
# def print_plot(config, reward_history, mean_reward_history):
#     print('max reward', max(reward_history))
#     print('mean reward', np.mean(reward_history))
#
#     print(f"Mean last 100: {np.mean(reward_history[-100:]):.1f}")
#
#     plt.plot(reward_history, label="reward_history", color="red")
#     plt.plot(mean_reward_history, label="mean_reward_history", color="blue")
#     plt.title("dqn_duel_per")
#     plt.text(0., 500, f'LR = {config.learning_rate:.4f}, gamma = {config.gamma:.3f}')
#     plt.text(0., 450, f'PER : ALPHA = {config.alpha:.2f}')
#
#     plt.text(0., 400, f'Seed = {config.seed:.2f}, target_up = {config.target_update_freq:.2f}')
#     plt.text(0., 350,f"Mean last 100: {np.mean(reward_history[-100:]):.1f}")
#
#
#     plt.ylim(-400, 600)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def log_params(config):
    # with mlflow.start_run(run_name='dqn_duel_tree'):
        mlflow.log_params({
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "epsilon_start": config.epsilon_start,
            "epsilon_end": config.epsilon_end,
            "epsilon_decay": config.epsilon_decay,
            "buffer_size": config.buffer_size,
            "batch_size": config.batch_size,
            "n_episodes": config.n_episodes,
            "max_step": config.max_step,
            "target_update_freq": config.target_update_freq,
            "alpha": config.alpha,
            "beta_start": config.beta_start,
            "beta_increment": config.beta_increment,
            "seed": config.seed,
            "env_name": config.env_name
        })

def log_metrics(agent , episode, total_reward,reward_history: list[float]):
    mlflow.log_metrics({
        "episode_reward": total_reward,
        "epsilon": agent.epsilon,
        "grad": agent.last_grad_norm,
        "td error": agent.last_td_error,
        "mean_reward_100": np.mean(reward_history[-100:]) if reward_history else total_reward
    }, step=episode)
    mlflow.log_metric(
        "buffer_fill",
        agent.memory.size / agent.memory.capacity,
        step=episode
    )