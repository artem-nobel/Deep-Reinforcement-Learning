import numpy as np

from algorithms.actor_critic.logger import log_params, log_metrics

class Trainer:

    def __init__(self, env, agent, config):

        self.env = env
        self.agent = agent
        self.config = config



    def run_episode(self):

        state, _ = self.env.reset()

        done = False

        episode_reward = 0

        step = 0

        log_probs = []
        values = []
        rewards = []

        while not done and step < self.config.max_step:

            action, log_prob, value = self.agent.get_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)

            done = terminated or truncated

            log_probs.append(log_prob)

            values.append(value)

            rewards.append(reward)

            state = next_state

            episode_reward += reward

            step += 1

            if self.config.render:
                self.env.render()

        loss_info = self.agent.update(
            log_probs,
            values,
            rewards,
        )

        return episode_reward, loss_info

    def train(self):

            log_params(self.config)
            reward_history = []
            mean_reward_history = []

            running_reward = 0

            for episode in range(self.config.n_episodes):

                reward , loss_info = self.run_episode()

                reward_history.append(reward)
                mean_reward_history.append(np.mean(reward_history[-100:]))
                log_metrics(
                    self.agent,
                    episode,
                    reward,
                    reward_history
                )
                running_reward = (
                        0.05 * reward
                        +
                        0.95 * running_reward
                )

                if episode % self.config.log_interval == 0:

                    print(
                        f"Episode {episode:5d} | "
                        f"Reward {reward:8.2f} | "
                        f"Running {running_reward:8.2f}"
                    )

                if (
                        running_reward >
                        self.env.spec.reward_threshold
                ):
                    print("Environment solved.")
                    break

            return (
                    reward_history,
                    mean_reward_history,
                    )