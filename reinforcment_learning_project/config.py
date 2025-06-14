import os

import gymnasium as gym

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = f"{CURR_DIR}/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

# Environment Configuration
ENV_NAME = "FrozenLake-v1"
ENV_CONFIG = {
    "is_slippery": False
}

# Q-Learning Parameters
LEARNING_RATE = 0.08  # -> 0.08
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
TRAINING_EPISODES = 2000
MAX_STEPS = 100
TEST_EPISODES = 10

# Visualization Settings
PLOT_INTERVAL = 100  # Plot every N episodes
SAVE_PLOTS = True
