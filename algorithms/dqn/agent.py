from replay_buffer import PrioritizedReplayBuffer
from config import Config
from model import DuelingDQN

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import _to_tensor

class Agent:
    def __init__(self, config: Config, n_states=16, n_actions=4):
        self.last_td_error = 0.0
        self.last_grad_norm = 0.0
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = config.learning_rate
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.device = config.device
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay


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

        self.train_freq = config.train_freq
        self.step_counter = 0

    def save(self, path="model.pth"):
        torch.save(self.main_model.state_dict(), path)
        print(f"Saved to {path}")

    def load(self, path="model.pth"):
        self.main_model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.main_model.state_dict())
        print(f"Loaded from {path}")

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            self.main_model.eval()
            with torch.no_grad():
                state_tensor = _to_tensor(state).unsqueeze(0)
                q_value = self.main_model(state_tensor)
                return torch.argmax(q_value).item()

    def get_actions_batch(self, states, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions, size=len(states))
        else:
            self.main_model.eval()
            with torch.no_grad():
                states_tensor = torch.tensor(states, dtype=torch.float32)
                q_values = self.main_model(states_tensor)
                return torch.argmax(q_values, dim=1).cpu().numpy()

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

            state_tensor = torch.stack([_to_tensor(s) for s in states])
            new_state_tensor = torch.stack([_to_tensor(s) for s in new_states])
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

    def update_batch(self, states, actions, rewards, new_states, dones):
        self.main_model.train()

        for i in range(len(states)):
            self.memory.add((
                states[i],
                actions[i],
                rewards[i],
                new_states[i],
                dones[i]
            ))

        self.step_counter += 1

        if self.step_counter % self.train_freq == 0 and self.memory.size >= self.batch_size:
            batch, indices, weights = self.memory.sample(self.batch_size)

            states_b, actions_b, rewards_b, new_states_b, dones_b = zip(*batch)

            weight = torch.tensor(weights, dtype=torch.float32)

            state_tensor = torch.stack([_to_tensor(s) for s in states_b])
            new_state_tensor = torch.stack([_to_tensor(s) for s in new_states_b])
            action_tensor = torch.tensor(actions_b, dtype=torch.long)
            reward_tensor = torch.tensor(rewards_b, dtype=torch.float32)
            done_tensor = torch.tensor(dones_b, dtype=torch.float32)

            current_q = self.main_model(state_tensor)
            current_q = current_q[range(len(action_tensor)), action_tensor]

            with torch.no_grad():
                next_action = self.main_model(new_state_tensor).argmax(1).unsqueeze(1)
                next_q = self.target_model(new_state_tensor).gather(1, next_action).squeeze()

            target_q = reward_tensor + self.gamma * next_q * (1 - done_tensor)

            td_error = (target_q - current_q).detach().numpy()

            self.memory.update_priorities(indices, td_error)

            loss = (weight * self.loss_fn(current_q, target_q)).mean()

            self.optimizer.zero_grad()
            loss.backward()
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