import gymnasium as gym
import mlflow
from matplotlib import pyplot as plt
from algorithms.ppo.agent import Agent
from algorithms.ppo.config import Config
from algorithms.ppo.trainer import *
from algorithms.ppo.utils import get_env_info, set_seed
from pathlib import Path
import matplotlib.pyplot as plt


config = Config()
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment(f"{config.env_name},ppo_batch")


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
        reward_history,
        alpha=0.3,
        label="Reward"
    )

    plt.plot(
        mean_reward_history,
        label="Mean reward"
    )

    plt.title(
        "PPO"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Reward"
    )

    plt.text(
        0.02,
        0.95,
        f"Environment: {config.env_name}\n"
        f"Seed: {config.seed}",

        transform=plt.gca().transAxes,
        verticalalignment="top"
    )

    plt.grid(alpha=0.3)
    plt.legend()

    plot_dir = Path(
        "/Users/artemhorkov/PycharmProjects/Deep-Reinforcement-Learning/algorithms/ppo/plot"
    )

    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        plot_dir / f"ppo_seed_{config.seed}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    main()
