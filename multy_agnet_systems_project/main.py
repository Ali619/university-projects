import time

from agent import Agent
from config import NUM_AGENTS
from visualization import SimulationVisualizer


def print_simulation_info():
    print("\n=== Multi-Agent System Simulation ===")
    print(f"Number of agents: {NUM_AGENTS}")
    print("Each agent will try to reach the goal while avoiding collisions")
    print("The simulation will show:")
    print("1. Real-time agent positions")
    print("2. Statistics including:")
    print("   - Total steps taken")
    print("   - Number of collisions")
    print("   - Average distance to goal")
    print("\nStarting simulation in 3 seconds...")
    time.sleep(3)


def print_final_statistics(agents):
    print("\n=== Final Statistics ===")
    for agent in agents:
        stats = agent.get_statistics()
        print(f"\nAgent {stats['id']}:")
        print(f"Steps taken: {stats['steps_taken']}")
        print(f"Collisions: {stats['collisions']}")
        print(
            f"Final distance to goal: {stats['current_distance_to_goal']:.2f}")
        print(f"Path length: {stats['path_length']}")


def main():
    # Print initial information
    print_simulation_info()

    # Create agents
    agents = [Agent(i) for i in range(NUM_AGENTS)]

    # Create and run visualization
    visualizer = SimulationVisualizer()
    ani = visualizer.create_animation(agents)
    visualizer.show()

    # Print final statistics
    print_final_statistics(agents)


if __name__ == "__main__":
    main()
