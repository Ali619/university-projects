import random

import numpy as np
from config import *


class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.epsilon = INITIAL_EPSILON
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Exploration
        return np.argmax(self.q_table[state])  # Exploitation

    def update(self, state, action, reward, next_state):
        """Update Q-table using Q-learning formula."""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        # Q-learning formula
        new_value = (1 - self.learning_rate) * old_value + \
            self.learning_rate * (reward + self.discount_factor * next_max)

        self.q_table[state, action] = new_value
        return abs(new_value - old_value)  # Return loss for tracking

    def decay_epsilon(self):
        """Decay epsilon value."""
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

    def get_q_table(self):
        """Return the current Q-table."""
        return self.q_table
