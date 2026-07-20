from algorithms.common.sum_tree import SumTree

import numpy as np


class PrioritizedReplayBuffer:
    """
        Prioritized Experience Replay Buffer.

        Implements:
        - sampling proportional to TD-error priorities
        - importance sampling weights
        - beta annealing
        """

    def __init__(
            self,
            capacity,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
    ):
        self.tree = SumTree(capacity)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

    @property
    def size(self):
        return self.tree.size

    @property
    def capacity(self):
        return self.tree.capacity

    def add(self, transition):
        max_priority = max(
            self.tree.get_max_priority(),
            1.0,
        )

        self.tree.add(
            max_priority,
            transition,
        )

    def sample(self, batch_size):
        self.beta = min(
            1.0,
            self.beta + self.beta_increment,
        )

        batch, indices = self.tree.sample(batch_size)

        probs = []

        for idx in indices:
            # probability of sampling transition
            prob = (
                    self.tree.tree[idx]
                    / self.tree.total()
                    + 1e-8
            )

            probs.append(prob)

        probs = np.array(probs)

        # importance sampling weights
        weights = (
                          self.size * probs
                  ) ** (-self.beta)

        weights /= weights.max()

        return (
            batch,
            indices,
            weights,
        )

    def update_priorities(
            self,
            indices,
            td_errors,
    ):
        for idx, error in zip(indices, td_errors):
            priority = (
                               abs(error) + 1e-8
                       ) ** self.alpha

            self.tree.update(
                idx,
                priority,
            )