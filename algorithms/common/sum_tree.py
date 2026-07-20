import random

import numpy as np


class SumTree:
    """
    Binary SumTree for Prioritized Experience Replay.

    Stores priorities in leaf nodes and maintains cumulative sums
    in internal nodes for O(log N) updates and sampling.

    Used to sample transitions with probability proportional to priority.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

        self.write = 0
        self.size = 0

    def total(self):
        return self.tree[0]

    def get_max_priority(self):
        if self.size == 0:
            return 1.0

        return self.tree[-self.capacity:].max()

    def add(self, priority, data):
        # Insert data at current write position (cyclic buffer)

        idx = self.write + self.capacity - 1

        self.data[self.write] = data

        self.update(idx, priority)

        self.write += 1

        if self.write >= self.capacity:
            self.write = 0

        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, priority):
        # Update leaf and propagate delta up the tree
        change = priority - self.tree[idx]

        self.tree[idx] = priority

        self._propagate(idx, change)

    def _propagate(self, idx, change):
        # Traverse tree to find leaf where cumulative sum >= value
        # Used for sampling proportional to priority
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, value):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)

        return self._retrieve(
            right,
            value - self.tree[left],
        )

    def get(self, value):
        idx = self._retrieve(0, value)

        data_idx = idx - (self.capacity - 1)

        return (
            idx,
            self.tree[idx],
            self.data[data_idx],
        )

    def sample(self, batch_size):
        # Splits the overall priority into equal segments
        # and selects one value from each segment
        batch = []
        indices = []

        segment = self.total() / batch_size

        for i in range(batch_size):
            start = segment * i
            end = segment * (i + 1)

            value = random.uniform(start, end)

            idx, _, data = self.get(value)

            batch.append(data)
            indices.append(idx)

        return batch, indices