# Q-Learning

Lightweight README for the Q-learning exercises in this repository.

## Overview
This folder contains code and exercises implementing the Q-learning algorithm for discrete Markov Decision Processes (MDPs). The goal is to learn an action-value function Q(s, a) that approximates expected return and derive a policy that maximizes cumulative reward.

## Features
- Tabular Q-learning implementation
- Epsilon-greedy exploration
- Support for configurable learning rate (alpha), discount factor (gamma), and epsilon schedule
- Example environments and scripts for training and evaluation

## Requirements
- Python 3.7+
- Common libraries: numpy, gym (optional for OpenAI Gym environments)
Install required packages:
```bash
pip install -r requirements.txt
```

## Quick start
Train a Q-learning agent on a grid/world or Gym environment (adjust paths/names to match local scripts):
```bash
python train_q_learning.py --env FrozenLake-v1 --episodes 5000 --alpha 0.1 --gamma 0.99 --epsilon 0.1
```
Evaluate a trained agent:
```bash
python eval_q_learning.py --model models/q_table.npy --episodes 100
```

## Algorithm (high-level)
1. Initialize Q(s, a) arbitrarily (e.g., zeros)
2. For each episode:
    - Initialize state s
    - Repeat for each step:
      - Choose action a using epsilon-greedy policy derived from Q
      - Take action a, observe reward r and next state s'
      - Update:
         Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
      - s <- s'
      - If terminal, end episode

## Hyperparameters
- alpha (learning rate): controls update step size
- gamma (discount factor): trade-off between immediate and future rewards
- epsilon (exploration): probability of random action; often decayed over time
- episodes / max_steps: training length

## Tips
- Normalize rewards or clip if necessary for stability.
- Use a decaying epsilon schedule (e.g., epsilon = max(eps_min, eps_start * decay^episode)).
- For large state spaces, consider function approximation (DQN) instead of tabular Q-learning.
