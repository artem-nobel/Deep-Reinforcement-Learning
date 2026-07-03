import numpy as np

from agent_vec import Agent
from config import Config
import gymnasium as gym
import gymnasium as gym
from config import Config
from agent import Agent
from utils import get_env_info

config = Config()

env = gym.make(config.env_name)
n_states, n_actions = get_env_info(env)
env.close()

agent = Agent(config, n_states=n_states, n_actions=n_actions)
agent.load("my_model2.pth")

def visualize(max_step=500):
    """Визуализация одного эпизода"""

    vis_env = gym.make(config.env_name, render_mode="human")

    state, _ = vis_env.reset()

    done = False
    step = 0
    total_reward = 0

    while not done and step < max_step:
        vis_env.render()  # Отрисовка
        action = agent.get_action(state, training=False)  # Без epsilon-жадности
        state, reward, terminated, truncated, _ = vis_env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

    print(f"Visualization: Total reward = {total_reward}, Steps = {step}")
    vis_env.close()
    return total_reward
# config = Config()
#
# agent = Agent(config)
#
# agent.load("my_model2.pth")
all_rewards = []
for episode in range(10):
    total_reward = visualize(max_step=config.max_step)
    all_rewards.append(total_reward)
    print('mean reward = {}'.format(np.mean(all_rewards)))