#!/usr/bin/env python3
# Import necessary libraries and bugfixes, activate GPU for faster training
import gymnasium as gym
import os
import ale_py
import tensorflow as tf
from gymnasium.wrappers import AtariPreprocessing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import legacy as legacy_optimizers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.keras import mixed_precision
from rl.callbacks import Callback

# Mixed precision & logging
mixed_precision.set_global_policy('mixed_float16')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
gym.register_envs(ale_py)

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Hyperparameters
# Gamma: discount factor for past rewards
gamma = 0.99
# Alpha: learning rate
alpha = 0.00025
# Batch size: number of samples per gradient update
batch_size = 128
# Number of actions: for discrete action spaces
num_actions = 4
# Memory limit: maximum number of experiences to store
memory_limit = 1_000_000
# Training parameters
train_interval = 4
# lower = more stable but slower training
target_model_update = 10000
# Number of steps to populate the replay memory before training starts
warmup_steps = 50_000
# Total number of training steps
nb_steps = 2_000_000
# Epsilon-greedy with linear annealing (start -> end over nb steps)
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_test = 0.05
# How many steps to anneal epsilon over (can be tuned)
epsilon_decay_steps = 1_000_000
visualize = False
# Model saving interval to prevent data loss
save_interval=100_000


class Wrapper(gym.Wrapper):
    # Implement wrapper for compatibility issue with KerasRL
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode='human'):
        return self.env.render()


def env_setup():
    # Environment setup
    # Grayscale, frame skip of 4, screen size 84x84
    env = gym.make("ALE/Breakout-v5", frameskip=1)
    env = AtariPreprocessing(env, screen_size=84,
                             grayscale_obs=True, frame_skip=4)
    env = Wrapper(env)
    return env


def create_q_model():
    # Define NN for training
    model = keras.Sequential([
        layers.InputLayer(input_shape=(4, 84, 84)),
        layers.Permute((2, 3, 1)),
        layers.Conv2D(32, 8, strides=4, activation="relu"),
        layers.Conv2D(64, 4, strides=2, activation="relu"),
        layers.Conv2D(64, 3, strides=1, activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(num_actions, activation="linear", dtype="float32"),
    ])
    return model


class SaveEveryNSteps(Callback):
    # Callback to save every N steps
    def __init__(self, save_interval, file_prefix="dqn_weights"):
        super().__init__()
        self.save_interval = save_interval
        self.file_prefix = file_prefix
        self.step_number = 0

    def on_step_end(self, step, logs={}):
        self.step_number += 1
        if self.step_number % self.save_interval == 0:
            filename = f"policy-{self.step_number}.h5"
            self.model.save_weights(filename, overwrite=True)
            print(f"\n Saved model at step {self.step_number} as {filename}")


def main():
    # Start enviroment, network, replay memory, policy and agent
    env = env_setup()
    model = create_q_model()
    memory = SequentialMemory(limit=memory_limit, window_length=4)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=epsilon_start,
        value_min=epsilon_end,
        value_test=epsilon_test,
        nb_steps=epsilon_decay_steps
    )
    agent = DQNAgent(
        model=model,
        nb_actions=num_actions,
        memory=memory,
        gamma=gamma,
        batch_size=batch_size,
        nb_steps_warmup=warmup_steps,
        train_interval=train_interval,
        target_model_update=target_model_update,
        policy=policy,
    )
    optimizer = legacy_optimizers.Adam(learning_rate=alpha)
    agent.compile(optimizer, metrics=["mae"])

    print("Starting training.")
    agent.fit(env, nb_steps=nb_steps, visualize=visualize, verbose=1,
              callbacks=[SaveEveryNSteps(save_interval=save_interval)])
    agent.save_weights("policy.h5", overwrite=True)
    print("Training complete — model saved.")


if __name__ == "__main__":
    main()
