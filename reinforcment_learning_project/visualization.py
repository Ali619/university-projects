import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_directory(plot_dir):
    """Create directory for saving plots if it doesn't exist."""
    Path(plot_dir).mkdir(parents=True, exist_ok=True)


def plot_training_progress(rewards, losses, epsilon_values, plot_dir):
    """Plot training metrics over episodes."""
    plt.figure(figsize=(15, 5))

    # Plot rewards
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Average Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    # Plot losses
    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.title('Q-Learning Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    # Plot epsilon decay
    plt.subplot(1, 3, 3)
    plt.plot(epsilon_values)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_progress.png'))
    plt.close()


def plot_q_table(q_table, plot_dir):
    """Plot the Q-table as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(q_table, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Q-Table Heatmap')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.savefig(os.path.join(plot_dir, 'q_table.png'))
    plt.close()


def print_episode_stats(episode, total_reward, steps, epsilon):
    """Print detailed statistics for each episode."""
    print(f"\nEpisode {episode + 1}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Steps taken: {steps}")
    print(f"Epsilon: {epsilon:.3f}")
    print("-" * 50)


def print_test_results(episode, success, steps):
    """Print results for test episodes."""
    status = "✅ Success" if success else "❌ Failed"
    print(f"\nTest Episode {episode + 1}")
    print(f"Status: {status}")
    print(f"Steps taken: {steps}")
    print("-" * 50)
