from algorithms.common.n_step import NStepBuffer
from algorithms.a2c.logger import (
    log_params,
    log_metrics
)

import numpy as np


class Trainer:


    def __init__(
            self,
            env,
            agent,
            config
    ):

        self.env = env
        self.agent = agent
        self.config = config


    def run_episode(self):
        state, _ = self.env.reset()
        done = False
        episode_reward = 0
        step = 0
        loss_info = None

        n_step_buffer = NStepBuffer(
            n_step=self.config.n_step,
            gamma=self.config.gamma
        )
        while not done and step < self.config.max_step:
            action = self.agent.get_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            n_step_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            if n_step_buffer.ready():
                transition = n_step_buffer.pop()
                loss_info = self.agent.update(
                    state=transition["state"],
                    action=transition["action"],
                    reward=transition["reward"],
                    next_state=transition["next_state"],
                    done=transition["done"],
                    n_step=transition["n_step"]
                )

            state = next_state
            episode_reward += reward
            step += 1

            if self.config.render:
                self.env.render()

        while len(n_step_buffer) > 0:
            transition = n_step_buffer.pop()
            if transition is not None:
                loss_info = self.agent.update(
                    state=transition["state"],
                    action=transition["action"],
                    reward=transition["reward"],
                    next_state=transition["next_state"],
                    done=transition["done"],
                    n_step=transition["n_step"]
                )

        return episode_reward, loss_info

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

                self.env.spec.reward_threshold

            ):


                print(
                    "Environment solved."
                )

                break



        return (

            reward_history,

            mean_reward_history

        )

