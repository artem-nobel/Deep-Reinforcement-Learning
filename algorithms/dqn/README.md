# Deep Q-Network (DQN)

A PyTorch implementation of the **Deep Q-Network (DQN)** algorithm for solving the **LunarLander-v3** environment from Gymnasium.

This project demonstrates the evolution of value-based reinforcement learning by combining several improvements over the original DQN algorithm.

---

## Features

* Experience Replay
* Target Network
* Double DQN
* Dueling DQN
* Prioritized Experience Replay (PER)
* Gradient Clipping
* MLflow experiment tracking
* Modular project structure

---

## Algorithm

The agent learns an action-value function **Q(s, a)** by minimizing the Temporal Difference (TD) error.

### TD Target

```text
target = reward + γ · max Qtarget(next_state)
```

For terminal states:

```text
target = reward
```

### Loss

```text
L = MSELoss(Q(s,a), target)
```

The target network is updated periodically to stabilize training.

---

## Implemented Improvements

### Experience Replay

Transitions are stored in a replay buffer and sampled randomly during training, reducing sample correlation.

### Target Network

A separate target network provides stable TD targets during optimization.

### Double DQN

Action selection is performed using the online network, while action evaluation is performed using the target network, reducing overestimation bias.

### Dueling DQN

The network separately estimates the state value and action advantages before combining them into Q-values.

### Prioritized Experience Replay

Transitions with larger TD errors are sampled more frequently, improving learning efficiency.

---

## Network Architecture

```text
State
   │
   ▼
Feature Extractor
   │
   ▼
Value Stream      Advantage Stream
      │                 │
      └────────┬────────┘
               ▼
           Q-values
```

---

## Project Structure

```text
dqn/
│
├── agent.py
├── replay_buffer.py
     ├── sum_tree
├── model.py
├── trainer.py
├── config.py
├── logger.py
├── utils.py
├── main.py
└── README.md
```

---

## Environment

**LunarLander-v3**

Observation space:

* 8 continuous state variables

Action space:

* 4 discrete actions

---

## Logging

Training metrics are recorded using **MLflow**.

Tracked metrics include:

* Episode reward
* Mean reward
* TD loss
* Epsilon
* Gradient norm

---

## Installation

```bash
pip install torch gymnasium[box2d] numpy matplotlib mlflow
```

---

## Run

```bash
python main.py
```

---

## Future Improvements

* Noisy Networks
* Distributional DQN (C51)
* Rainbow DQN
* Multi-step TD Learning
* Soft Target Updates

---

## License

MIT License
