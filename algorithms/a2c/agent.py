import torch
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical

from algorithms.a2c.model import ActorCriticNetwork


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


        self.loss_fn = nn.SmoothL1Loss()


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


    def get_action(self, state):

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        )
        with torch.no_grad():
            logits, value = self.model(state)

        dist = Categorical(
            logits=logits
        )


        action = dist.sample()

        return (
            action.item()
        )


    def update(
            self,
            state,
            action,
            reward,
            next_state,
            done,
            n_step
    ):

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        )


        next_state = torch.tensor(
            next_state,
            dtype=torch.float32,
            device=self.device
        )


        action = torch.tensor(
            action,
            device=self.device
        )


        logits, value = self.model(state)

        dist = Categorical(
            logits=logits
        )


        log_prob = dist.log_prob(
            action
        )


        with torch.no_grad():

            _, next_value = self.model(
                next_state
            )

            next_value = next_value.squeeze(-1)

            if done:
                target = torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=self.device
                )

            else:
                target = (
                    reward
                    +
                    (self.gamma ** n_step)
                    *
                    next_value
                )


        value = value.squeeze()

        advantage = (
            target - value
        )



        entropy = dist.entropy().mean()
        entropy_coef = 0.05


        # actor_loss = (
        #     -log_prob *
        #     advantage.detach()
        #     # advantage_clipped.detach()
        #     -
        #     entropy_coef
        #     *
        #     entropy
        # )
        # actor_loss = -log_prob * advantage.detach() - entropy_coef * entropy
        actor_loss = (-log_prob * advantage.detach()
                      # ).mean()
                      )


        critic_loss = self.loss_fn(
            value,
            target.detach()

        )


        loss = (
            actor_loss
            +
            # 0.2
            0.5
            # 1
            *
            critic_loss
        )


        self.optimizer.zero_grad()

        loss.backward()


        # grad_norm = torch.nn.utils.clip_grad_norm_(
        #     self.model.parameters(),
        #     1.0
        # )



        self.optimizer.step()



        self.last_loss = loss.item()
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        self.last_grad_norm = 0
        # (
            # grad_norm.item())


        self.last_value = value.item()
        self.last_target = target.item()
        self.last_advantage = advantage.item()
        self.last_value_error = abs(
            target.item()
            -
            value.item()
        )


        return {"loss": loss.item(),

            "actor_loss": actor_loss.item(),

            "critic_loss": critic_loss.item(),

            # "grad_norm": grad_norm.item()
            # "grad_norm": grad_norm.item()

        }
