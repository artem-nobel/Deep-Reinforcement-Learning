import gymnasium as gym
import mlflow
from matplotlib import pyplot as plt
from algorithms.a2c_batch.agent import Agent
from algorithms.a2c_batch.config import Config
from algorithms.a2c_batch.trainer import *
from algorithms.a2c_batch.utils import get_env_info, set_seed


config = Config()
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment(f"{config.env_name},a2c_n_step")


def main():

    set_seed(config.seed)

    envs = [
        gym.make(config.env_name)
        for _ in range(config.num_envs)
    ]

    # for env in envs:
    #     state, _ = env.reset(seed=config.seed)
    for i, env in enumerate(envs):
        env.reset(
            seed=config.seed + i
        )

    n_states, n_actions = get_env_info(
        config.env_name
    )

    agent = Agent(
        config=config,
        n_states=n_states,
        n_actions=n_actions,
    )

    trainer = Trainer(
        envs=envs,
        agent=agent,
        config=config,
    )

    with mlflow.start_run():
        reward_history, mean_reward_history = trainer.train()

    for env in envs:
       env.close()

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        reward_history
    )
    plt.plot(
        mean_reward_history
    )

    plt.title(
        "a2c_n_step"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Reward"
    )

    plt.grid()

    plt.show()

if __name__ == "__main__":

    main()
