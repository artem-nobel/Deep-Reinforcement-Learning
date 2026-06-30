
from replay_buffer import PrioritizedReplayBuffer
from config import Config
from model import DuelingDQN

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import gymnasium as gym




# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")




class Agent:
    def __init__(self, config: Config, n_states=16, n_actions=4):
        self.last_td_error = 0.0
        self.last_grad_norm = 0.0
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = config.learning_rate
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.max_step = config.max_step
        self.visualize_every = config.visualize_every
        self.env_name = config.env_name


        self.memory = PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            alpha=config.alpha,
            beta=config.beta_start,
            beta_increment=config.beta_increment,
        )
        self.batch_size = config.batch_size

        self.main_model = DuelingDQN(self.n_states, self.n_actions)
        self.target_model = DuelingDQN(self.n_states, self.n_actions)
        self.target_model.load_state_dict(self.main_model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(
            self.main_model.parameters(),
            lr=self.lr,
        )
        self.loss_fn = nn.MSELoss(reduction='none')

        self.target_update_count = 0
        self.target_update_max_step = config.target_update_freq

    def visualize(self, max_step=500):
        """Визуализация одного эпизода"""

        vis_env = gym.make(self.env_name, render_mode="human")

        state, _ = vis_env.reset()

        done = False
        step = 0
        total_reward = 0

        while not done and step < max_step:
            vis_env.render()  # Отрисовка
            action = self.get_action(state, training=False)  # Без epsilon-жадности
            state, reward, terminated, truncated, _ = vis_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        print(f"Visualization: Total reward = {total_reward}, Steps = {step}")
        vis_env.close()

    def save_model(self, path="model.pth"):
        torch.save(self.main_model.state_dict(), path)
        print(f"Saved to {path}")

    def load_model(self, path="model.pth"):
        self.main_model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.main_model.state_dict())
        print(f"Loaded from {path}")

    def _state_to_tensor(self, state):
        return torch.tensor(state, dtype=torch.float32)

    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            self.main_model.eval()
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state).unsqueeze(0)
                q_value = self.main_model(state_tensor)
                return torch.argmax(q_value).item()

    def update(self, state, action, reward, new_state, done):
        self.main_model.train()

        self.memory.add(
            (
                state,
                action,
                reward,
                new_state,
                done,
            )
        )

        if self.memory.size >= self.batch_size:

            batch, indices, weights = self.memory.sample(
                self.batch_size
            )

            states, actions, rewards, new_states, dones = zip(*batch)

            weight = torch.tensor(
            weights,
            dtype=torch.float32,
            )

            state_tensor = torch.stack([self._state_to_tensor(s) for s in states])
            new_state_tensor = torch.stack([self._state_to_tensor(s) for s in new_states])
            action_tensor = torch.tensor(actions, dtype=torch.long)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            done_tensor = torch.tensor(dones, dtype=torch.float32)

            current_q = self.main_model(state_tensor)
            current_q = current_q[range(len(action_tensor)), action_tensor]

            with torch.no_grad():
                next_action = self.main_model(new_state_tensor).argmax(1).unsqueeze(1)
                next_q = self.target_model(new_state_tensor).gather(1, next_action).squeeze()
            target_q = reward_tensor + self.gamma * next_q * (1 - done_tensor)

            td_error = (target_q - current_q).detach().numpy()


            self.memory.update_priorities(
                indices,
                td_error,
            )

            loss = (weight * self.loss_fn(current_q, target_q)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            # логирование градиента
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.main_model.parameters(),
                max_norm=1.0
            )

            self.last_td_error = np.mean(np.abs(td_error))
            self.last_grad_norm = grad_norm.item()
            self.optimizer.step()

            self.target_update_count += 1
            if self.target_update_count >= self.target_update_max_step:
                self.target_model.load_state_dict(self.main_model.state_dict())
                self.target_update_count = 0

    def train(self, env, n_episodes=100, seed=None):
        success_history = []
        reward_history = []
        mean_reward_history = []
        for episode in range(n_episodes):
            state, info = env.reset(seed=seed + episode)

            total_reward = 0
            step = 0
            done = False
            while not done and step <= self.max_step:
                action = self.get_action(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.update(state, action, reward, new_state, done)

                state = new_state
                total_reward += reward
                step += 1

            mlflow.log_metrics({
                "episode_reward": total_reward,
                "epsilon": self.epsilon,
                "grad": self.last_grad_norm,
                "td error": self.last_td_error,
                "mean_reward_100": np.mean(reward_history[-100:]) if reward_history else total_reward
            }, step=episode)
            mlflow.log_metric(
                "buffer_fill",
                self.memory.size / self.memory.capacity,
                step=episode
            )

            print("Episode:", episode, "Total reward:", total_reward, "epsilon:", self.epsilon)
            print(f"Mean last 100: {np.mean(reward_history[-100:]):.1f}")

            mean_reward_history.append(np.mean(reward_history[-100:]))

            reward_history.append(total_reward)
            success_history.append(total_reward > 0)

            # if episode % self.visualize_every == 0 and episode > 0:
            #     print(f"\n--- Visualization at episode {episode} ---")
            #     self.visualize(max_step=self.max_step)

            self.epsilon = max(0.01, self.epsilon * 0.997)

        return success_history, reward_history, mean_reward_history






