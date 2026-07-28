from collections import deque


class NStepBuffer:

    def __init__(self, n_step: int, gamma: float):

        self.n_step = n_step
        self.gamma = gamma

        self.buffer = deque()

    def add(
            self,
            state,
            action,
            reward,
            next_state,
            done,
            **kwargs,
    ):

        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

        transition.update(kwargs)

        self.buffer.append(transition)

    def ready(self):

        return (
            len(self.buffer) >= self.n_step
            or (
                    len(self.buffer) > 0
                    and self.buffer[-1]["done"]
            )
        )

    def pop(self):
        reward = 0.0
        next_state = None
        done = False
        used_steps = 0

        transitions_to_use = []
        for i in range(min(self.n_step, len(self.buffer))):
            transition = self.buffer[i]
            transitions_to_use.append(transition)
            reward += (self.gamma ** i) * transition["reward"]
            next_state = transition["next_state"]
            done = transition["done"]
            used_steps += 1

            if done:
                break

        # delete all transition
        # for _ in range(len(transitions_to_use)):
        #     self.buffer.popleft()

        # delete first transition
        self.buffer.popleft()

        first = transitions_to_use[0]
        result = dict(first)
        result.update({
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "n_step": used_steps,
        })

        return result

    def clear(self):

        self.buffer.clear()

    def __len__(self):

        return len(self.buffer)