from algorithms.dqn.replay_buffer import PrioritizedReplayBuffer

def test_add():
    buffer = PrioritizedReplayBuffer(100)
    buffer.add((1,2,3,4,5))
    print('add',buffer.size)
    assert buffer.size == 1

def test_capacity():
    buffer = PrioritizedReplayBuffer(5)

    for i in range(10):
        buffer.add(i)
    print(buffer.tree.tree)
    assert buffer.size == 5

def test_sample():

    buffer = PrioritizedReplayBuffer(100)

    for i in range(20):
        buffer.add(i)

    batch, indices, weights = buffer.sample(8)

    assert len(batch) == 8
    assert len(indices) == 8
    assert len(weights) == 8

def test_priority():
    buffer = PrioritizedReplayBuffer(100)

    # buffer.add((1, 2, 3, 4, 5))
    for i in range(10):
        buffer.add(i)
    print(buffer.tree.get_max_priority())


def test_cyclic_buffer():
    buffer = PrioritizedReplayBuffer(3)

    buffer.add("A")
    buffer.add("B")
    buffer.add("C")
    buffer.add("D")

    assert buffer.size == 3

def test_tree_size():

    buffer = PrioritizedReplayBuffer(8)

    assert len(buffer.tree.tree) == 15


