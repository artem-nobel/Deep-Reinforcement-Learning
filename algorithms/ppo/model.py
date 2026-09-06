import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.Tanh(),
        )

        self.actor = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, state):
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value