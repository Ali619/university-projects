import gymnasium as gym
from config import *


class FrozenLakeEnv:
    def __init__(self):
        self.env = gym.make(ENV_NAME, **ENV_CONFIG)
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

    def reset(self):
        """Reset the environment and return initial state."""
        return self.env.reset()

    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)

    def render(self):
        """Render the current state of the environment."""
        self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    def get_state_size(self):
        """Return the number of possible states."""
        return self.state_size

    def get_action_size(self):
        """Return the number of possible actions."""
        return self.action_size
