# Monte Carlo Actor-Critic

A PyTorch implementation of the **Monte Carlo Actor-Critic** algorithm for solving the **LunarLander-v3** environment.

This project demonstrates the transition from REINFORCE to Actor-Critic by introducing a value function (Critic) that estimates state values and reduces the variance of policy gradient updates.

---

## Features

* PyTorch implementation
* Monte Carlo policy updates
* Shared Actor-Critic network
* State-value baseline
* SmoothL1 (Huber) loss
* Gradient clipping
* MLflow experiment tracking
* Modular project structure

---

## Algorithm

The Actor updates the policy using Monte Carlo returns, while the Critic estimates the state-value function.

### Monte Carlo Return

```text
Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ...
```

### Advantage

```text
advantage = Gₜ - V(s)
```

### Actor Loss

```text
L_actor = -log π(a|s) · advantage
```

### Critic Loss

```text
L_critic = SmoothL1(V(s), Gₜ)
```

### Total Loss

```text
L = L_actor + L_critic
```

---

## Network Architecture

```text
                 State
                   │
                   ▼
          Shared Feature Extractor
        Linear(128) + ReLU
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
      Actor Head         Critic Head
   Action Logits          Value V(s)
```

---

## Training Process

```text
Episode
   │
   ▼
Collect trajectory
   │
   ▼
Compute Monte Carlo Returns
   │
   ▼
Compute Advantage
   │
   ▼
Update Actor & Critic
```

The policy is updated **after the entire episode has finished**.

---

## Project Structure

```text
actor_critic/
│
├── agent.py
├── trainer.py
├── model.py
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
* Mean reward (last 100 episodes)
* Total loss
* Actor loss
* Critic loss
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

## Difference from TD Actor-Critic

This implementation uses **Monte Carlo returns**, meaning that updates are performed only after the episode ends.

In contrast, TD Actor-Critic performs updates after every interaction using bootstrapped value estimates.

---

## Future Improvements

* One-Step TD Actor-Critic
* Advantage Actor-Critic (A2C)
* A3C
* PPO
* Generalized Advantage Estimation (GAE)

---

## License

MIT License
