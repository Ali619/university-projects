# Multi-Agent System Project Documentation

## Overview
This project simulates a multi-agent system where multiple agents (robots/entities) navigate through a shared environment to reach a common goal while avoiding collisions with each other. The system demonstrates fundamental concepts of multi-agent systems, including:
- Autonomous agent behavior
- Collision avoidance
- Path planning
- Real-time visualization
- Performance metrics tracking

## System Components

### 1. Environment
- The environment is a 2D grid of size 20x20 (configurable via `ENV_SIZE`)
- A common goal is set at the top-right corner (ENV_SIZE-1, ENV_SIZE-1)
- The system supports multiple agents (default: 5 agents)

### 2. Agent System (`agent.py`)
Each agent in the system has the following characteristics:
- Unique ID for identification
- Current position (x, y coordinates)
- Movement capabilities
- Collision detection
- Statistics tracking

Key agent behaviors:
- **Movement**: Agents move towards the goal using a simple directional movement
- **Collision Avoidance**: Agents detect and avoid collisions with other agents
- **Path Tracking**: Each agent maintains a history of its movement
- **Statistics**: Agents track:
  - Steps taken
  - Number of collisions
  - Distance to goal
  - Path length

### 3. Visualization System (`visualization.py`)
The project includes a real-time visualization system with two main components:

1. **Main Simulation View**:
   - Shows agent positions in real-time
   - Displays the goal position
   - Grid-based environment visualization

2. **Statistics Dashboard**:
   - Tracks and displays:
     - Total steps taken by all agents
     - Total collisions
     - Average distance to goal
   - Updates in real-time

### 4. Configuration (`config.py`)
The system is highly configurable through parameters:
- Environment size
- Number of agents
- Animation settings
- Visualization colors and labels

## How It Works

1. **Initialization**:
   - The system creates the specified number of agents
   - Each agent is placed at a random position in the lower-left quadrant
   - The visualization system is initialized

2. **Simulation Loop**:
   - Each agent attempts to move towards the goal
   - Movement is calculated based on the direction to the goal
   - Collision detection occurs before each movement
   - If a collision is detected, the agent's collision counter increases
   - Agent positions and statistics are updated
   - The visualization is updated in real-time

3. **Termination**:
   - The simulation runs for a specified number of frames
   - Final statistics are displayed for each agent
   - The visualization shows the complete path history

## Key Features

1. **Autonomous Behavior**:
   - Each agent operates independently
   - Agents make decisions based on their current state and environment

2. **Collision Avoidance**:
   - Simple but effective collision detection
   - Agents can detect and respond to potential collisions

3. **Real-time Visualization**:
   - Live tracking of agent movements
   - Real-time statistics updates
   - Clear visual representation of the environment

4. **Performance Metrics**:
   - Comprehensive statistics tracking
   - Multiple performance indicators
   - Easy-to-understand visualization of metrics

## Usage

To run the simulation:
1. Ensure all dependencies are installed (from `requirements.txt`)
2. Run `main.py`
3. The simulation will start automatically
4. Watch the real-time visualization
5. Review the final statistics when the simulation completes

## Technical Implementation

The project uses:
- NumPy for numerical computations
- Matplotlib for visualization
- Object-oriented programming for clean code organization
- Animation for real-time updates

This project serves as an excellent example of a basic multi-agent system, demonstrating key concepts in a visual and interactive way. It's particularly useful for understanding:
- Multi-agent coordination
- Collision avoidance strategies
- Real-time system visualization
- Performance metrics in multi-agent systems
