import torch
import torch.optim as optim
from torch.distributions import Categorical

from model import ActorCriticNetwork


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

        self.loss_fn = torch.nn.SmoothL1Loss()

        self.last_loss = 0
        self.last_actor_loss = 0
        self.last_critic_loss = 0
        self.last_grad_norm = 0

    def save(
            self,
            path="actor_critic_montecarlo.pth"
    ):

        # torch.save(
        #     self.model.state_dict(),
        #     path
        # )
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        })

        print(f"Saved to {path}")

    def load(
            self,
            path="actor_critic_montecarlo.pth"
    ):

        self.model.load_state_dict(
            torch.load(
                path,
                map_location=self.device
            )
        )

        print(f"Loaded from {path}")

    def get_action(self,state):

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        logits, value = self.model(state)

        probs = torch.softmax(
            logits,
            dim=-1
        )

        distribution = Categorical(
            probs
        )

        action = distribution.sample()

        return (
            action.item(),
            distribution.log_prob(action),
            value.squeeze()
        )

    def compute_returns(
            self,
            rewards,
    ):

        discounted_reward = 0

        returns = []

        for reward in reversed(rewards):

            discounted_reward = (
                    reward
                    + self.gamma * discounted_reward
            )

            returns.insert(
                0,
                discounted_reward
            )

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )

        # normalize returns for more stable training
        returns = (
                returns
                - returns.mean()
        ) / (
                returns.std()
                + 1e-8
        )

        return returns

    def update(
            self,
            log_probs,
            values,
            rewards,
    ):

        returns = self.compute_returns(
            rewards
        )

        actor_losses = []

        critic_losses = []

        for log_prob, value, target in zip(log_probs,values,returns):

            advantage = (
                    target
                    - value.detach()
            )

            actor_losses.append(
                -log_prob * advantage
            )

            critic_losses.append(
                self.loss_fn(
                    value,
                    target
                )
            )
        actor_loss = torch.stack(
            actor_losses
        ).sum()

        critic_loss = torch.stack(
            critic_losses
        ).sum()

        loss = (
                actor_loss
                + critic_loss
        )

        self.optimizer.zero_grad()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        self.last_loss = loss.item()
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        self.last_grad_norm = grad_norm.item()


        self.optimizer.step()

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "grad_norm": grad_norm.item()
        }