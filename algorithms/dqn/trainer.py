import gymnasium as gym
import numpy as np

from gymnasium.vector import AsyncVectorEnv


def make_env(env_name, seed):
    def _init():
        env = gym.make(env_name)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return _init


def create_vector_env(env_name, n_envs, seed):
    """
    Создает AsyncVectorEnv один раз.
    """

    return AsyncVectorEnv(
        [
            make_env(env_name, seed + i)
            for i in range(n_envs)
        ]
    )


def train_episode(agent, env, config, seed):

    state, _ = env.reset(seed=seed)

    total_reward = 0
    step = 0
    done = False

    while not done and step < config.max_step:

        action = agent.get_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        agent.update(
            state,
            action,
            reward,
            next_state,
            done,
        )

        state = next_state

        total_reward += reward
        step += 1

    return total_reward, step


def train_episode_vectorized(
    agent,
    envs,
    config,
    seed,
):
    """
    Один логический эпизод обучения.

    Все среды работают одновременно.
    """

    states, _ = envs.reset(seed=seed)

    rewards_sum = np.zeros(config.n_envs)

    step = 0

    while step < config.max_step:

        actions = agent.get_actions_batch(
            states,
            training=True,
        )

        (
            next_states,
            rewards,
            terminated,
            truncated,
            infos,
        ) = envs.step(actions)

        dones = np.logical_or(
            terminated,
            truncated,
        )

        agent.update_batch(
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

        rewards_sum += rewards

        states = next_states

        step += 1

    return rewards_sum.mean(), step