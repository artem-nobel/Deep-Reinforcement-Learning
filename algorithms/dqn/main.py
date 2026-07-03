from config import Config
from agent import Agent
from loger import *
from trainer import *
import random
import gymnasium as gym
import mlflow
from utils import *

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("lunar_lander_dqn_dueling")



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
    agent = Agent(
        config=config,
        n_states=n_states,
        n_actions=n_actions
    )
    with mlflow.start_run(run_name="dqn_duel_tree"):

        log_params(config)
        success_history = []
        reward_history = []
        mean_reward_history = []
        for episode in range(config.n_episodes):

            total_reward, step = train_episode(
                agent,
                env,
                config,
                seed + episode
            )

            print("Episode:", episode, "Total reward:", total_reward, "epsilon:", agent.epsilon)
            print(f"Mean last 100: {np.mean(reward_history[-100:]):.1f}")

            reward_history.append(total_reward)
            success_history.append(total_reward > 0)
            mean_reward_history.append(np.mean(reward_history[-100:]))

            log_metrics(agent, episode, total_reward ,reward_history)

            agent.decay_epsilon()

    agent.save("my_model2.pth")
    mlflow.pytorch.log_model(agent.main_model, "model")

    plt.figure()
    plt.plot(reward_history, label="reward_history", color="red")
    plt.plot(mean_reward_history, label="mean_reward_history", color="blue")
    plt.title("dqn_duel_per")
    plt.legend()
    plt.grid(True)

    plt.savefig("training_plot.png")
    mlflow.log_artifact("training_plot.png")
    plt.close()

    print_plot(config, reward_history, mean_reward_history)

