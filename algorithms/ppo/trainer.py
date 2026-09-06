import numpy as np

from algorithms.ppo.buffer import RollOutBuffer

from algorithms.ppo.logger import (
    log_params,
    log_metrics
)


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

        self.num_envs = len(envs)

        self.buffer = RollOutBuffer(

            gamma=config.gamma,

            gae_lambda=config.gae_lambda,

            rollout_size=config.rollout_size,

            num_envs=self.num_envs
        )

    def run_episode(self):

        states = []

        for env in self.envs:

            state, _ = env.reset()

            states.append(state)

        states = np.asarray(
            states,
            dtype=np.float32
        )

        episode_rewards = np.zeros(
            self.num_envs,
            dtype=np.float32
        )

        completed_rewards = []

        loss_info = None

        for step in range(
            self.config.max_steps
        ):

            actions, old_log_probs, values = (
                self.agent.get_action(states)
            )

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

                episode_rewards[i] += reward

                next_states.append(
                    next_state
                )

                rewards.append(
                    reward
                )

                dones.append(
                    done
                )

                if done:

                    completed_rewards.append(
                        episode_rewards[i]
                    )

                    episode_rewards[i] = 0.0

                    reset_state, _ = env.reset()

                    next_states[-1] = reset_state

            next_states = np.asarray(
                next_states,
                dtype=np.float32
            )

            rewards = np.asarray(
                rewards,
                dtype=np.float32
            )

            dones = np.asarray(
                dones,
                dtype=np.float32
            )

            self.buffer.add(

                states=states,

                actions=actions.detach().cpu().numpy(),

                rewards=rewards,

                next_states=next_states,

                log_probs=(
                    old_log_probs
                    .detach()
                    .cpu()
                    .numpy()
                ),

                values=(
                    values
                    .detach()
                    .cpu()
                    .numpy()
                ),

                dones=dones
            )

            states = next_states

            if self.buffer.ready():

                last_values = self.agent.get_value(
                    states
                )

                (
                    states_batch,
                    actions_batch,
                    old_log_probs_batch,
                    advantages_batch,
                    returns_batch
                ) = self.buffer.get_data(
                    last_values
                )

                advantages_batch = (
                    advantages_batch
                    -
                    advantages_batch.mean()
                ) / (
                    advantages_batch.std()
                    + 1e-8
                )

                for epoch in range(
                    self.config.ppo_epochs
                ):

                    indices = np.random.permutation(
                        len(states_batch)
                    )


                    for start in range(
                        0,
                        len(states_batch),
                        self.config.batch_size
                    ):

                        end = (
                            start
                            +
                            self.config.batch_size
                        )

                        batch_indices = (
                            indices[start:end]
                        )

                        minibatch_states = (
                            states_batch[
                                batch_indices
                            ]
                        )

                        minibatch_actions = (
                            actions_batch[
                                batch_indices
                            ]
                        )

                        minibatch_old_log_probs = (
                            old_log_probs_batch[
                                batch_indices
                            ]
                        )

                        minibatch_advantages = (
                            advantages_batch[
                                batch_indices
                            ]
                        )

                        minibatch_returns = (
                            returns_batch[
                                batch_indices
                            ]
                        )

                        loss_info = self.agent.update(

                            states=minibatch_states,

                            actions=minibatch_actions,

                            old_log_probs=(
                                minibatch_old_log_probs
                            ),

                            advantages=(
                                minibatch_advantages
                            ),

                            returns=(
                                minibatch_returns
                            )
                        )

                self.buffer.clear()

        if completed_rewards:

            reward = np.mean(
                completed_rewards
            )

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

        running_reward = 0.0

        for episode in range(
            self.config.n_episodes
        ):

            reward, loss_info = (
                self.run_episode()
            )

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

                    f"Mean100 "
                    f"{np.mean(reward_history[-100:]):8.2f}"
                )

            if (
                running_reward
                >
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
