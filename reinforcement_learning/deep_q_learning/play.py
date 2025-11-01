#!/usr/bin/env python3
# Import necessary libraries and bugfixes, activate GPU for faster training
import gymnasium as gym
import os
import ale_py
import tensorflow as tf
from gymnasium.wrappers import AtariPreprocessing
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

# Bugfixes and GPU setup
gym.register_envs(ale_py)
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel("ERROR")
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_CUDA_PLUGIN"] = "0"
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"


# Hyperparameters
# Gamma: discount factor for past rewards
gamma = 0.99
# Alpha: learning rate
alpha = 0.00025
# Number of actions: for discrete action spaces
num_actions = 4
# Number of frames stacked together as input
window_length = 4
# Memory limit: maximum number of experiences to store
memory_limit = 1_000_000
# lower = more stable but slower training
target_model_update = 1
# Number of steps to populate the replay memory before training starts
warmup_steps = 0
# Total number of training steps
nb_steps = 1
# Number of episodes to test
nb_episodes = 5
# visualization parameters
visualize = True
render_mode = "human"


class Wrapper(gym.Wrapper):
    # Implement wrapper for compatibility issue with KerasRL
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        return observation

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, mode='human'):
        return self.env.render()


def env_setup(render_mode="human"):
    # Create environment with human render mode for visualization
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                             frame_skip=1, noop_max=30)
    env = Wrapper(env)
    return env


def create_q_model():
    # Define NN for training
    model = keras.Sequential([
        layers.InputLayer(input_shape=(4, 84, 84)),
        layers.Reshape((84, 84, 4)),
        layers.Conv2D(32, 8, strides=4, activation="relu"),
        layers.Conv2D(64, 4, strides=2, activation="relu"),
        layers.Conv2D(64, 3, strides=1, activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(num_actions, activation="linear", dtype="float32"),
    ])
    return model


def main():
    # Start environment, network, replay memory, policy and agent
    env = env_setup(render_mode=render_mode)
    model = create_q_model()
    policy = GreedyQPolicy()
    memory = SequentialMemory(limit=memory_limit, window_length=window_length)
    agent = DQNAgent(model=model, nb_actions=num_actions, memory=memory,
                     policy=policy, nb_steps_warmup=warmup_steps,
                     target_model_update=target_model_update, gamma=gamma)
    agent.compile(keras.optimizers.Adam(learning_rate=alpha), metrics=["mae"])
    agent.load_weights("policy.h5")
    agent.test(env, nb_episodes=nb_episodes, visualize=visualize)


if __name__ == "__main__":
    main()
