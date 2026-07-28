import gymnasium as gym
import mlflow
from matplotlib import pyplot as plt
from algorithms.a2c.agent import Agent
from algorithms.a2c.config import Config
from algorithms.a2c.trainer import *
from algorithms.a2c.utils import get_env_info, set_seed


config = Config()
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment(f"{config.env_name},a2c_n_step")


def main():

    set_seed(config.seed)

    env = gym.make(
        config.env_name,
        render_mode="human"
        if config.render
        else None
    )

    env.reset(
        seed=config.seed
    )

    n_states, n_actions = get_env_info(
        env
    )

    agent = Agent(
        config=config,
        n_states=n_states,
        n_actions=n_actions,
    )

    trainer = Trainer(
        env=env,
        agent=agent,
        config=config,
    )

    with mlflow.start_run():
        reward_history, mean_reward_history = trainer.train()

    env.close()

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        reward_history
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
