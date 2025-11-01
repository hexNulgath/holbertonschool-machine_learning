# deep_q_learning

## Project intent
Repository contains a Deep Q-Learning implementation with two main entry points:
- `train.py` — trains the agent (long-running process).
- `play.py` — runs a trained agent for evaluation/play.

## Dependencies
There are many compativility issues with the used libraries so it is important to use the exact version of the following modules
keras-rl2==1.0.4
gymnasium[atari]==0.29.1
tensorflow==2.15.0
keras==2.15.0
numpy==1.25.2
Pillow==10.3.0
h5py==3.11.0
autorom[accept-rom-license]

## Training notes
- Training DQN is compute- and time-intensive: expect many hours (or days) depending on environment complexity, network size, and hardware.
- External issues (OOM, environment disconnects, driver resets) can crash runs unpredictably. To mitigate this, checkpoint regularly.

## Checkpointing strategy
- Save model + optimizer + replay buffer periodically (every X iterations / episodes). Example policy:
    - Save every X=1000 training iterations (configurable).
    - Also save a "best" checkpoint when validation or average reward improves.
- Recommended saved items:
    - model state_dict
    - optimizer state_dict
    - current iteration / episode
    - epsilon / exploration schedule state
    - replay buffer snapshot (if feasible)

## Usage
- Train: `python train.py`
- Play: `python play.py`

