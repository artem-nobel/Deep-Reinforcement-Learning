import torch

from algorithms.a2c_n_step.logger import (
    log_params,
    log_metrics
)

import numpy as np


class Trainer:


    def __init__(
            self,
            envs: list,
            agent,
            config
    ):

        self.envs = envs
        self.agent = agent
        self.config = config


    def run_episode(self):
        states = []

        for env in self.envs :
            state, _ = env.reset()
            states.append(state)

        states = np.array(
            states,
            dtype=np.float32
        )

        episode_rewards = np.zeros(
            len(self.envs),
            dtype=np.float32
        )

        completed_rewards = []

        loss_info = None

        for step in range(self.config.max_steps):

            actions, log_probs, values = self.agent.get_action(states)

            next_states = []
            rewards = []
            dones = []

            for i, env in enumerate(self.envs):
                next_state, reward, terminated, truncated, _ = (
                    env.step(
                        actions[i].item()
                    )
                )

                done = (
                        terminated
                        or
                        truncated
                )

                next_states.append(
                    next_state
                )

                rewards.append(
                    reward
                )

                dones.append(
                    done
                )

                episode_rewards[i] += reward

                if done:
                    completed_rewards.append(
                        episode_rewards[i]
                    )

                    episode_rewards[i] = 0.0

                    next_state, _ = env.reset()

                    next_states[-1] = next_state

            next_states = np.array(
                next_states,
                dtype=np.float32
            )

            loss_info = self.agent.update(

                states=states,

                actions=actions,

                rewards=rewards,

                next_states=next_states,

                dones=dones,

                log_probs=log_probs,

                values=values,
            )

            states = next_states

        if completed_rewards:
            reward = np.mean(completed_rewards)
        else:
            reward = 0.0

        return (
            reward,
            loss_info
        )

    def train(self):


        log_params(
            self.config
        )


        reward_history = []

        mean_reward_history = []


        running_reward = 0



        for episode in range(
            self.config.n_episodes
        ):



            reward, loss_info = self.run_episode()



            reward_history.append(
                reward
            )



            mean_reward_history.append(

                np.mean(
                    reward_history[-100:]
                )

            )



            log_metrics(

                self.agent,

                episode,

                reward,

                reward_history,

                loss_info

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

                    f"Running {running_reward:8.2f} | "

                    f"Mean100 {np.mean(reward_history[-100:]):8.2f}"

                )



            if (

                running_reward >

                self.envs[0].spec.reward_threshold

            ):


                print(
                    "Environment solved."
                )

                break



        return (

            reward_history,

            mean_reward_history

        )

