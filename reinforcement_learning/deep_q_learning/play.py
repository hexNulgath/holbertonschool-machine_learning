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
os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ["SDL_RENDER_DRIVER"] = "software"


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
        # Return only observation for keras-rl compatibility
        observation, info = self.env.reset(**kwargs)

        # Auto-FIRE to start the game to prevent waiting state
        # triggering noop actions
        try:
            meanings = self.env.unwrapped.get_action_meanings()
            if "FIRE" in meanings:
                fire_action = meanings.index("FIRE")
                for i in range(2):
                    res = self.env.step(fire_action)
                    if isinstance(res, tuple) and len(res) == 5:
                        obs2, reward, terminated, truncated, info2 = res
                        done = terminated or truncated
                    else:
                        obs2, reward, done, info2 = res
                    if done:
                        observation, info = self.env.reset(**kwargs)
                        break
        except Exception:
            pass

        # Initialize lives tracking
        try:
            self.last_lives = self._get_lives()
        except Exception:
            self.last_lives = None

        return observation

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            observation, reward, done, info = result

        # Auto-FIRE on life loss to prevent waiting state
        try:
            current_lives = self._get_lives()
            if self.last_lives is not None and current_lives is not None and current_lives < self.last_lives:
                try:
                    meanings = self.env.unwrapped.get_action_meanings()
                    if "FIRE" in meanings:
                        fire_action = meanings.index("FIRE")
                        for i in range(2):
                            _res = self.env.step(fire_action)
                            if isinstance(_res, tuple):
                                if len(_res) == 5:
                                    observation, reward2, terminated2, truncated2, info = _res
                                    done = terminated2 or truncated2
                                else:
                                    observation, reward2, done, info = _res
                            if done:
                                break
                except Exception:
                    pass
            self.last_lives = current_lives
        except Exception:
            pass

        return observation, reward, done, info

    def _get_lives(self):
        # help function to get current number of lives for triggering auto-FIRE
        try:
            if hasattr(self.env.unwrapped, "ale") and hasattr(
                 self.env.unwrapped.ale, "lives"):
                return int(self.env.unwrapped.ale.lives())
        except Exception:
            pass
        try:
            if hasattr(self.env, "last_info") and isinstance(
                 self.env.last_info, dict) and "lives" in self.env.last_info:
                return int(self.env.last_info["lives"])
        except Exception:
            pass
        return None

    def render(self, mode='human'):
        return self.env.render()


def env_setup(render_mode="human"):
    # Create environment with human render mode for visualization
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                             frame_skip=4)
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
