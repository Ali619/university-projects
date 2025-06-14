# Reinforcement Learning Project: Frozen Lake Environment

## Overview
This project implements a Q-Learning algorithm to solve the Frozen Lake environment, which is a classic reinforcement learning problem. The goal is to train an agent to navigate from a starting point to a goal while avoiding holes in the ice.

## Core Components

### 1. Environment (`environment.py`)
- The Frozen Lake environment is a grid world where:
  - The agent starts at position S
  - Must reach the goal G
  - Must avoid holes H
  - Can move on frozen surface F
- The environment provides:
  - State observations
  - Rewards
  - Action space (up, down, left, right)

### 2. Q-Learning Agent (`agent.py`)
The agent implements the Q-Learning algorithm with these key components:

#### Q-Table
- A matrix that stores the expected future rewards for each state-action pair
- Initialized with zeros
- Updated through learning

#### Key Methods:
1. `choose_action(state)`:
   - Uses epsilon-greedy strategy
   - With probability ε: explores (random action)
   - With probability 1-ε: exploits (best known action)

2. `update(state, action, reward, next_state)`:
   - Updates Q-values using the Q-learning formula:
   ```
   Q(s,a) = (1-α)Q(s,a) + α[r + γ max(Q(s',a'))]
   ```
   Where:
   - α (alpha) = learning rate
   - γ (gamma) = discount factor
   - r = reward
   - s' = next state
   - a' = next action

3. `decay_epsilon()`:
   - Gradually reduces exploration rate
   - Helps transition from exploration to exploitation

### 3. Training Process (`main.py`)
The training process follows these steps:

1. **Initialization**:
   - Creates environment and agent
   - Sets up visualization tools

2. **Training Loop**:
   - Runs for specified number of episodes
   - Each episode:
     - Resets environment
     - Agent takes actions until goal or failure
     - Updates Q-table
     - Tracks metrics (rewards, losses, epsilon)

3. **Testing Phase**:
   - Evaluates trained agent
   - Calculates success rate
   - Shows performance metrics

### 4. Visualization (`visualization.py`)
Provides visual feedback through:
- Training rewards over time
- Q-Learning loss
- Epsilon decay
- Q-table heatmap

## How It Works

1. **Learning Process**:
   - Agent starts with no knowledge (Q-table = 0)
   - Explores environment through trial and error
   - Updates Q-values based on rewards received
   - Gradually learns optimal path to goal

2. **Exploration vs Exploitation**:
   - Starts with high exploration (ε ≈ 1)
   - Gradually shifts to exploitation
   - Balances learning new paths vs using known good paths

3. **Reward Structure**:
   - Positive reward for reaching goal
   - Negative reward (or no reward) for falling in holes
   - Small negative reward for each step (encourages efficiency)

## Key Parameters (in `config.py`)
- Learning rate (α): How quickly the agent updates its knowledge
- Discount factor (γ): How much future rewards are valued
- Epsilon decay: How quickly exploration decreases
- Training episodes: Number of learning iterations
- Maximum steps: Steps allowed per episode

## Running the Project
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python main.py`
3. View results in console and `plots` directory

This project demonstrates fundamental reinforcement learning concepts through a practical implementation. The agent learns to solve the Frozen Lake problem through experience, gradually improving its strategy from random exploration to optimal path finding.