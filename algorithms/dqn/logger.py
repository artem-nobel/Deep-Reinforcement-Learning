import mlflow
import numpy as np


def log_params(config):
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