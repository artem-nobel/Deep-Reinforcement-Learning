from algorithms.dqn.agent import *

def test_agent_initialization():
    config = Config()

    agent = Agent(config)

    assert agent.n_states == 16
    assert agent.n_actions == 4
    assert agent.epsilon == config.epsilon_start

def test_epsilon_decay():
    config = Config()

    agent = Agent(config)

    for _ in range(100000):
        agent.decay_epsilon()

    assert agent.epsilon == config.epsilon_end

def test_epsilon_decreases():
    config = Config()

    agent = Agent(config)

    old = agent.epsilon

    agent.decay_epsilon()

    assert agent.epsilon < old

def test_random_action():
    config = Config()

    agent = Agent(config)

    agent.epsilon = 1.0

    state = np.zeros(agent.n_states)

    actions = []

    for _ in range(100):
        actions.append(agent.get_action(state))

    assert len(set(actions)) > 1

# def test_greedy_action():
#     config = Config()
#
#     agent = Agent(config)
#
#     agent.epsilon = 0
#
#     agent.main_model = agent.load("my_model2")
#     state = np.zeros(agent.n_states)
#
#     action = agent.get_action(state)
#
#     assert action == 1

def test_get_actions_batch_range():
    config = Config()

    agent = Agent(config)

    agent.epsilon = 1

    states = np.zeros((64, agent.n_states))

    actions = agent.get_actions_batch(states)

    assert np.all(actions >= 0)
    assert np.all(actions < agent.n_actions)

def test_update_adds_transition():
    config = Config()

    config.batch_size = 100

    agent = Agent(config)

    state = np.zeros(agent.n_states)

    agent.update(
        state,
        0,
        1,
        state,
        False
    )

    assert agent.memory.size == 1

