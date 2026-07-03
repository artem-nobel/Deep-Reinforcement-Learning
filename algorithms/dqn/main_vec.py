from config import Config
from agent import Agent
from logger import *
from trainer import *

from matplotlib import pyplot as plt

import torch
import random
import gymnasium as gym
import numpy as np
import mlflow

from utils import *

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("my_self_ml_flow")


if __name__ == "__main__":

    config = Config()

    seed = config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    env = gym.make(config.env_name)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    n_states, n_actions = get_env_info(env)

    env.close()


    envs = create_vector_env(
        config.env_name,
        config.n_envs,
        seed
    )

    agent = Agent(
        config=config,
        n_states=n_states,
        n_actions=n_actions
    )

    with mlflow.start_run(run_name="dqn_duel_async_vector"):

        log_params(config)

        reward_history = []
        mean_reward_history = []
        success_history = []

        for episode in range(config.n_episodes):

            total_reward, step = train_episode_vectorized(
                agent,
                envs,
                config,
                seed + episode
            )

            reward_history.append(total_reward)

            success_history.append(total_reward > 0)

            mean_reward_history.append(
                np.mean(reward_history[-100:])
            )

            log_metrics(
                agent,
                episode,
                total_reward,
                reward_history
            )

            print(
                f"Episode {episode:4d} | "
                f"Reward {total_reward:8.2f} | "
                f"Mean100 {mean_reward_history[-1]:7.2f} | "
                f"Epsilon {agent.epsilon:.3f}"
            )

            agent.decay_epsilon()


    envs.close()

    agent.save("my_model2.pth")

    mlflow.pytorch.log_model(
        agent.main_model,
        "model"
    )

    plt.figure()

    plt.plot(
        reward_history,
        label="reward"
    )

    plt.plot(
        mean_reward_history,
        label="mean100"
    )

    plt.grid(True)

    plt.legend()

    plt.savefig("training_plot.png")

    mlflow.log_artifact("training_plot.png")

    plt.close()

    print_plot(
        config,
        reward_history,
        mean_reward_history
    )