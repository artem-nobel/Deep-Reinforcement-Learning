import torch
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical

from algorithms.a2c_n_step.model import ActorCriticNetwork


class Agent:

    def __init__(self, config, n_states, n_actions):

        self.gamma = config.gamma
        self.device = config.device

        self.model = ActorCriticNetwork(
            n_states,
            n_actions
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.critic_coef = config.critic_coef

        # self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss()

        self.last_loss = 0
        self.last_actor_loss = 0
        self.last_critic_loss = 0
        self.last_grad_norm = 0

        self.last_value = 0
        self.last_target = 0
        self.last_advantage = 0
        self.last_value_error = 0

    def save(self, path="a2c_n_step.pth"):

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)


    def load(self, path="a2c_n_step.pth"):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )



    def get_action(self, states):

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        logits, values = self.model(states)

        probs = torch.softmax(
            logits,
            dim=-1
        )

        distribution = Categorical(probs)

        actions = distribution.sample()

        return (
            actions,
            distribution.log_prob(actions),
            values.squeeze(-1)
        )

    def update(
            self,
            states,
            actions,
            rewards,
            next_states,
            dones,
            log_probs,
            values,
    ):
        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            _, next_values = self.model(
                next_states
            )

            next_values = next_values.squeeze(-1)

            targets = (
                    rewards
                    +
                    self.gamma
                    *
                    next_values
                    *
                    (1 - dones)
            )

        advantages = (
                targets - values
        )

        actor_loss = -(
                log_probs
                *
                advantages.detach()
        ).mean()

        critic_loss = self.loss_fn(
            values,
            targets
        )

        loss = (
                actor_loss
                +
                self.critic_coef
                *
                critic_loss
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.last_loss = loss.item()

        self.last_actor_loss = actor_loss.item()

        self.last_critic_loss = critic_loss.item()

        self.last_value = values.mean().item()

        self.last_target = targets.mean().item()

        self.last_advantage = advantages.mean().item()

        self.last_value_error = (
            advantages.abs().mean().item()
        )

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }
