import torch
import torch.optim as optim

from torch import nn
from torch.distributions import Categorical

from algorithms.ppo.model import ActorCriticNetwork


class Agent:

    def __init__(
            self,
            config,
            n_states,
            n_actions
    ):

        self.device = config.device

        self.clip_epsilon = config.clip_epsilon
        self.critic_coef = config.critic_coef

        self.model = ActorCriticNetwork(
            n_states,
            n_actions
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

        self.loss_fn = nn.MSELoss()

        self.last_loss = 0.0
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

        self.last_grad_norm = 0.0
        self.last_value = 0.0
        self.last_target = 0.0
        self.last_advantage = 0.0
        self.last_value_error = 0.0

    def get_action(self, states):

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        logits, values = self.model(states)

        distribution = Categorical(
            logits=logits
        )

        actions = distribution.sample()

        log_probs = distribution.log_prob(
            actions
        )

        values = values.squeeze(-1)

        return (
            actions,
            log_probs,
            values
        )

    def get_value(self, states):

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():

            _, values = self.model(states)

            values = values.squeeze(-1)

        return values.cpu().numpy()

    def update(
            self,
            states,
            actions,
            old_log_probs,
            advantages,
            returns
    ):

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        old_log_probs = torch.as_tensor(
            old_log_probs,
            dtype=torch.float32,
            device=self.device
        )

        advantages = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=self.device
        )

        returns = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )


        logits, values = self.model(states)

        values = values.squeeze(-1)

        distribution = Categorical(
            logits=logits
        )

        new_log_probs = distribution.log_prob(
            actions
        )

        ratio = torch.exp(
            new_log_probs - old_log_probs
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon
        )

        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        value_loss = self.loss_fn(
            values,
            returns
        )

        loss = (
            policy_loss
            +
            self.critic_coef * value_loss
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.last_loss = loss.item()
        self.last_actor_loss = policy_loss.item()
        self.last_critic_loss = value_loss.item()

        return {

            "loss": loss.item(),

            "actor_loss":
                policy_loss.item(),

            "critic_loss":
                value_loss.item(),

        }

