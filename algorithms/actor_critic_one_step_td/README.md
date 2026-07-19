# One-Step Actor-Critic (TD)

A simple implementation of the **One-Step Actor-Critic** reinforcement learning algorithm using **PyTorch** and **Gymnasium**.

This project demonstrates the transition from Monte-Carlo policy gradient methods (REINFORCE with baseline) to temporal-difference (TD) learning. The agent is trained on the **LunarLander-v3** environment and uses a shared neural network with separate Actor and Critic heads.

---

## Features

* PyTorch implementation
* Gymnasium environment
* Shared Actor-Critic neural network
* One-step TD learning
* Huber (SmoothL1) loss for Critic
* Gradient clipping
* MLflow experiment tracking
* Configurable hyperparameters
* Modular project structure

---

## Algorithm

Unlike REINFORCE, which waits until the end of an episode to compute returns, this implementation updates the network **after every environment step** using Temporal Difference (TD) learning.

### Critic target

```text
target = r + γ · V(s')
```

If the episode terminates:

```text
target = r
```

### Advantage

```text
advantage = target - V(s)
```

### Actor loss

```text
L_actor = -log π(a|s) · advantage
```

### Critic loss

```text
L_critic = SmoothL1(V(s), target)
```

### Total loss

```text
L = L_actor + L_critic
```

---

## Project Structure

```text
actor_critic/
│
├── agent.py          # Agent and TD update
├── trainer.py        # Training loop
├── model.py          # Actor-Critic network
├── config.py         # Hyperparameters
├── logger.py         # MLflow logging
├── utils.py          # Utilities
├── main.py           # Entry point
└── README.md
```

---

## Network Architecture

```
State
   │
   ▼
Linear(128)
   │
 ReLU
   │
 ├───────────────┐
 │               │
 ▼               ▼
Actor          Critic
(logits)       V(s)
```

The network consists of a shared feature extractor followed by two independent output heads:

* **Actor** predicts action logits.
* **Critic** estimates the value of the current state.

---

## Training

The training loop follows the sequence:

```text
State
   │
   ▼
Actor selects action
   │
   ▼
Environment step
   │
   ▼
Critic estimates next value
   │
   ▼
Compute TD target
   │
   ▼
Compute Advantage
   │
   ▼
Update Actor and Critic
```

The network parameters are updated **after every interaction with the environment**.

---

## Environment

* LunarLander-v3

Observation Space:

* 8 continuous state variables

Action Space:

* 4 discrete actions

---

## Logging

Training statistics are recorded using **MLflow**.

Tracked metrics include:

* Episode reward
* Mean reward (last 100 episodes)
* Total loss
* Actor loss
* Critic loss
* Gradient norm

---

## Requirements

```text
Python 3.10+
PyTorch
Gymnasium
NumPy
MLflow
Matplotlib
```

Install dependencies:

```bash
pip install torch gymnasium[box2d] numpy matplotlib mlflow
```

---

## Run

```bash
python main.py
```

---

## Notes

This project implements a **one-step TD Actor-Critic** algorithm.

Although it shares the same Actor-Critic architecture as **Advantage Actor-Critic (A2C)**, it performs parameter updates after every environment step instead of collecting multi-step rollouts.

This implementation is intended as an educational bridge between:

```text
REINFORCE
      ↓
REINFORCE + Baseline
      ↓
One-Step Actor-Critic (TD)
      ↓
A2C
      ↓
A3C
      ↓
PPO
```

---

## Future Improvements

* n-step returns
* Advantage Actor-Critic (A2C)
* Entropy regularization
* Vectorized environments
* Parallel data collection
* Generalized Advantage Estimation (GAE)
* Proximal Policy Optimization (PPO)

---

## License

MIT License
