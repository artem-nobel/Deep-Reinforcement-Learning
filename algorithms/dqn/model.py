import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.features_nn = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.value_nn = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.advantage_nn = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        feature = self.features_nn(x)
        value = self.value_nn(feature)
        advantage = self.advantage_nn(feature)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
