import gymnasium as gym
import mlflow
from matplotlib import pyplot as plt
from agent import Agent
from config import Config
from trainer import Trainer
from utils import get_env_info
from utils import set_seed


mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("lunar_lander_actor_critic")


def main():

    config = Config()

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


    reward_history, mean_reward_history = trainer.train()

    env.close()

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        reward_history
    )

    plt.title(
        "Actor-Critic training"
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
