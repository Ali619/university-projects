import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from config import *


class SimulationVisualizer:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.scat = self.ax1.scatter([], [], c=AGENT_COLOR)
        self.goal_plot, = self.ax1.plot(
            GOAL[0], GOAL[1], marker='o', color=GOAL_COLOR, label=GOAL_LABEL)

        # Statistics plot
        self.stats_lines = {}
        self.stats_data = {
            'steps': [],
            'collisions': [],
            'distance': []
        }

    def init_plot(self):
        # Main simulation plot
        self.ax1.set_xlim(0, ENV_SIZE)
        self.ax1.set_ylim(0, ENV_SIZE)
        self.ax1.set_title("Agent Positions")
        self.ax1.grid(True)
        self.ax1.legend()

        # Statistics plot
        self.ax2.set_title("Simulation Statistics")
        self.ax2.set_xlabel("Time Step")
        self.ax2.grid(True)

        return self.scat, self.goal_plot

    def update_plot(self, frame, agents):
        # Update agent positions
        positions = np.array([agent.position for agent in agents])
        self.scat.set_offsets(positions)

        # Update statistics
        total_steps = sum(agent.steps_taken for agent in agents)
        total_collisions = sum(agent.collisions for agent in agents)
        avg_distance = np.mean([agent.get_distance_to_goal()
                               for agent in agents])

        self.stats_data['steps'].append(total_steps)
        self.stats_data['collisions'].append(total_collisions)
        self.stats_data['distance'].append(avg_distance)

        # Plot statistics
        self.ax2.clear()
        self.ax2.plot(self.stats_data['steps'], label='Total Steps')
        self.ax2.plot(self.stats_data['collisions'], label='Total Collisions')
        self.ax2.plot(self.stats_data['distance'],
                      label='Avg Distance to Goal')
        self.ax2.legend()
        self.ax2.grid(True)

        return self.scat, self.goal_plot

    def create_animation(self, agents):
        return animation.FuncAnimation(
            self.fig,
            lambda frame: self.update_plot(frame, agents),
            frames=ANIMATION_FRAMES,
            init_func=self.init_plot,
            interval=ANIMATION_INTERVAL,
            blit=True,
            repeat=False
        )

    def show(self):
        plt.tight_layout()
        plt.show()

    def save_animation(self, filename):
        self.ani.save(filename, writer='pillow')
