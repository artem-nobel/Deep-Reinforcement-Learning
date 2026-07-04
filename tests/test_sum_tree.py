import numpy as np
import pytest

from algorithms.dqn.replay_buffer import SumTree


@pytest.mark.parametrize("capacity", [2, 4, 8, 16, 32])
def test_tree_initialization(capacity):
    tree = SumTree(capacity)

    assert tree.capacity == capacity
    assert tree.size == 0
    assert tree.write == 0
    assert len(tree.tree) == 2 * capacity - 1
    assert len(tree.data) == capacity
    assert tree.total() == 0


def test_add_one_element():
    tree = SumTree(8)

    tree.add(1.0, "A")
    tree.add(2.0, "B")
    tree.add(3.0, "C")

    assert tree.size == 3
    assert tree.write == 3
    assert tree.total() == 6.0


def test_total_priority():
    tree = SumTree(8)

    tree.add(1.0, "A")
    tree.add(2.0, "B")
    tree.add(3.0, "C")

    assert tree.total() == 6.0


def test_get_max_priority():
    tree = SumTree(8)

    tree.add(1.0, "A")
    tree.add(5.0, "B")
    tree.add(2.0, "C")

    assert tree.get_max_priority() == 5.0


def test_update_priority():
    tree = SumTree(8)

    tree.add(1.0, "A")

    leaf_index = tree.capacity - 1

    tree.update(leaf_index, 10.0)

    assert tree.total() == 10.0
    assert tree.get_max_priority() == 10.0


def test_get_returns_correct_data():
    tree = SumTree(4)

    tree.add(1.0, "A")
    tree.add(2.0, "B")
    tree.add(3.0, "C")

    idx, priority, data = tree.get(0.5)

    assert data == "A"
    assert priority == 1.0


def test_size_never_exceeds_capacity():
    tree = SumTree(3)

    tree.add(1, "A")
    tree.add(1, "B")
    tree.add(1, "C")
    tree.add(1, "D")
    tree.add(1, "E")

    assert tree.size == 3


def test_write_pointer_wraps():
    tree = SumTree(3)

    tree.add(1, "A")
    tree.add(1, "B")
    tree.add(1, "C")

    assert tree.write == 0

    tree.add(1, "D")

    assert tree.write == 1


def test_old_data_is_overwritten():
    tree = SumTree(2)

    tree.add(1, "A")
    tree.add(1, "B")
    tree.add(1, "C")

    assert "C" in tree.data
    assert "B" in tree.data


def test_total_after_overwrite():
    tree = SumTree(2)

    tree.add(1, "A")
    tree.add(2, "B")

    assert tree.total() == 3

    tree.add(4, "C")

    assert tree.total() == 6


def test_sample_always_returns_valid_data():
    tree = SumTree(8)

    for i in range(8):
        tree.add(i + 1, i)

    batch, indices = tree.sample(4)

    assert len(batch) == 4
    assert len(indices) == 4

    for idx in indices:
        assert 7 <= idx <= 14


def test_all_priorities_positive():
    tree = SumTree(4)

    tree.add(1, "A")
    tree.add(2, "B")
    tree.add(3, "C")
    tree.add(4, "D")

    leaves = tree.tree[-tree.capacity:]

    assert np.all(leaves > 0)

def test_update_changes_total():
    tree = SumTree(4)

    tree.add(1, "A")
    tree.add(1, "B")

    old_total = tree.total()

    tree.update(tree.capacity - 1, 5)

    assert tree.total() != old_total

