class RollOutBuffer:

    def __init__(self, gamma, gae_lambda, rollout_size):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_size = rollout_size
        self.buffer = []

    def ready(self):
        return len(self.buffer) >= self.rollout_size

    def add(self, state, action, reward, next_state, log_prob, value, done):

        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "log_prob": log_prob,
            "value": value,
        }

        self.buffer.append(transition)

    def compute_gae(self):

        advantages = [0.0] * len(self.buffer)

        gae = 0.0

        for t in reversed(range(len(self.buffer))):

            reward = self.buffer[t]["reward"]
            value = self.buffer[t]["value"]
            done = self.buffer[t]["done"]

            if t == len(self.buffer) - 1:
                next_value = 0.0
            else:
                next_value = self.buffer[t + 1]["value"]

            delta = (
                    reward
                    + self.gamma * next_value * (1 - done)
                    - value
            )

            gae = (
                    delta
                    + self.gamma * self.gae_lambda * (1 - done) * gae
            )
            print(t,gae)
            advantages[t] = gae

        return advantages


