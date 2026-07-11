import mlflow
import numpy as np


def log_params(config):

    mlflow.log_params({
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "n_episodes": config.n_episodes,
        "max_step": config.max_step,
        "seed": config.seed,
        "env_name": config.env_name
    })


def log_metrics(agent, episode, reward, reward_history, loss_info=None):
    if not mlflow.active_run():
        mlflow.start_run()

    metrics = {
        "episode_reward": reward,
        "loss": agent.last_loss,
        "actor_loss": agent.last_actor_loss,
        "critic_loss": agent.last_critic_loss,
        "grad_norm": agent.last_grad_norm,
        "mean_reward_100": np.mean(reward_history[-100:])
    }

    if loss_info:
        metrics.update(loss_info)

    mlflow.log_metrics(metrics, step=episode)