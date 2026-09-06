import numpy as np


class RollOutBuffer:

    def __init__(
            self,
            gamma: float,
            gae_lambda: float,
            rollout_size: int,
            num_envs: int
    ):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_size = rollout_size
        self.num_envs = num_envs

        self.buffer = []

    def ready(self):

        return len(self.buffer) >= self.rollout_size

    def add(
            self,
            states,
            actions,
            rewards,
            next_states,
            log_probs,
            values,
            dones
    ):

        self.buffer.append({

            "state": np.asarray(states),
            "action": np.asarray(actions),
            "reward": np.asarray(rewards),
            "next_state": np.asarray(next_states),
            "log_prob": np.asarray(log_probs),
            "value": np.asarray(values),
            "done": np.asarray(dones),

        })

    def compute_gae(self, last_values):

        n_steps = len(self.buffer)

        advantages = np.zeros(
            (n_steps, self.num_envs),
            dtype=np.float32
        )

        gae = np.zeros(
            self.num_envs,
            dtype=np.float32
        )

        for t in reversed(range(n_steps)):

            rewards = self.buffer[t]["reward"]
            values = self.buffer[t]["value"]
            dones = self.buffer[t]["done"]

            if t == n_steps - 1:

                next_values = np.asarray(
                    last_values,
                    dtype=np.float32
                )

            else:

                next_values = self.buffer[t + 1]["value"]

            delta = (
                rewards
                +
                self.gamma
                * next_values
                * (1.0 - dones)
                -
                values
            )

            gae = (
                delta
                +
                self.gamma
                * self.gae_lambda
                * (1.0 - dones)
                * gae
            )

            advantages[t] = gae

        values = np.array(
            [
                transition["value"]
                for transition in self.buffer
            ],
            dtype=np.float32
        )

        returns = advantages + values

        return advantages, returns

    def get_data(self, last_values):

        advantages, returns = (
            self.compute_gae(last_values)
        )

        states = np.array(
            [
                transition["state"]
                for transition in self.buffer
            ],
            dtype=np.float32
        )

        actions = np.array(
            [
                transition["action"]
                for transition in self.buffer
            ],
            dtype=np.int64
        )

        old_log_probs = np.array(
            [
                transition["log_prob"]
                for transition in self.buffer
            ],
            dtype=np.float32
        )

        # [steps, envs, ...]
        # ->
        # [steps * envs, ...]

        states = states.reshape(
            -1,
            states.shape[-1]
        )

        actions = actions.reshape(-1)

        old_log_probs = old_log_probs.reshape(-1)

        advantages = advantages.reshape(-1)

        returns = returns.reshape(-1)

        return (
            states,
            actions,
            old_log_probs,
            advantages,
            returns
        )

    def clear(self):

        self.buffer.clear()



