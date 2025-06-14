import random

import numpy as np
from config import ENV_SIZE, GOAL


class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.position = np.array([random.randint(0, ENV_SIZE//2),
                                  random.randint(0, ENV_SIZE//2)])
        self.path_history = [self.position.copy()]  # Track movement history
        self.collisions = 0
        self.steps_taken = 0

    def move_towards_goal(self, others):
        direction = GOAL - self.position
        # Move in positive or negative direction
        direction = np.sign(direction)

        # Check for collisions with other agents
        next_pos = self.position + direction
        collision_occurred = False

        for other in others:
            if other.id != self.id and np.array_equal(other.position, next_pos):
                collision_occurred = True
                self.collisions += 1
                return collision_occurred

        # Update position if no collision
        self.position = next_pos
        self.path_history.append(self.position.copy())
        self.steps_taken += 1
        return collision_occurred

    def get_distance_to_goal(self):
        return np.linalg.norm(GOAL - self.position)

    def get_statistics(self):
        return {
            'id': self.id,
            'steps_taken': self.steps_taken,
            'collisions': self.collisions,
            'current_distance_to_goal': self.get_distance_to_goal(),
            'path_length': len(self.path_history)
        }
