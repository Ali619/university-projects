import numpy as np
from agent import QLearningAgent
from config import *
from environment import FrozenLakeEnv
from visualization import *


def train_agent():
    """Train the Q-learning agent with detailed progress tracking."""
    print("🚀 Starting Q-Learning Training...")
    print(f"Environment: {ENV_NAME}")
    print(f"Training Episodes: {TRAINING_EPISODES}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Discount Factor: {DISCOUNT_FACTOR}")
    print("-" * 50)

    # Initialize environment and agent
    env = FrozenLakeEnv()
    agent = QLearningAgent(env.get_state_size(), env.get_action_size())

    # Setup visualization
    setup_plot_directory(PLOT_DIR)

    # Training metrics
    rewards_history = []
    losses_history = []
    epsilon_history = []

    # Training loop
    for episode in range(TRAINING_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0

        for step in range(MAX_STEPS):
            # Choose and perform action
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update Q-table and track loss
            loss = agent.update(state, action, reward, next_state)
            episode_loss += loss
            total_reward += reward

            state = next_state

            if done:
                break

        # Update metrics
        rewards_history.append(total_reward)
        losses_history.append(episode_loss / (step + 1))
        epsilon_history.append(agent.epsilon)

        # Print progress
        if (episode + 1) % 10 == 0:
            print_episode_stats(episode, total_reward, step + 1, agent.epsilon)

        # Plot progress periodically
        if (episode + 1) % PLOT_INTERVAL == 0:
            plot_training_progress(
                rewards_history, losses_history, epsilon_history, PLOT_DIR)

        agent.decay_epsilon()

    # Final plots
    plot_training_progress(rewards_history, losses_history,
                           epsilon_history, PLOT_DIR)
    plot_q_table(agent.get_q_table(), PLOT_DIR)

    return agent


def test_agent(agent):
    """Test the trained agent with detailed results."""
    print("\n🧪 Testing Trained Agent...")
    env = FrozenLakeEnv()
    success_count = 0

    for episode in range(TEST_EPISODES):
        state, _ = env.reset()
        steps = 0

        for step in range(MAX_STEPS):
            action = np.argmax(agent.q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if done:
                success = reward > 0
                success_count += int(success)
                print_test_results(episode, success, steps)
                break

    success_rate = (success_count / TEST_EPISODES) * 100
    print(f"\n📊 Final Results:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Successful Episodes: {success_count}/{TEST_EPISODES}")


def main():
    """Main function to run the entire training and testing process."""
    print("🎮 Frozen Lake Q-Learning Project")
    print("=" * 50)

    # Train the agent
    agent = train_agent()

    # Test the agent
    test_agent(agent)

    print("\n✨ Project completed! Check the 'plots' directory for visualizations.")


if __name__ == "__main__":
    main()
