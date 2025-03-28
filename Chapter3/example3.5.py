import numpy as np
import random

class GridworldEnvironment():
    def __init__(self):
        self.value_function = np.zeros((5, 5))
    
    # dynamics of the MDP p(s',r|s,a)
    def mdp_dynamics(self, state, action):
        '''
        Deterministic dynamics of the MDP, mapping state s and action a
        to state s' and reward r. A state is defined by the tuple (row, col).
        If s is on the boundary of the grid and the action is such that the
        next state is outside, then s' = s and reward = -1.
        '''
        
        row, col = state
        
        # Special states
        if (row, col) == (0, 1):   # State A
            return ((4, 1), 10)
        elif (row, col) == (0, 3): # State B
            return ((2, 3), 5)

        # Regular states
        if action == 'n':
            next_state = (max(row - 1, 0), col)
            reward = -1 if row == 0 else 0      # Penalize boundary collision
        elif action == 's':
            next_state = (min(row + 1, 4), col)
            reward = -1 if row == 4 else 0
        elif action == 'e':
            next_state = (row, min(col + 1, 4))
            reward = -1 if col == 4 else 0
        elif action == 'w':
            next_state = (row, max(col - 1, 0))
            reward = -1 if col == 0 else 0
        else:
            raise ValueError("Invalid action!")

        return (next_state, reward)

class Agent():
    def __init__(self):
        self.actions = ['n','s','e','w']
    
agent = Agent()
env = GridworldEnvironment()

gamma = 0.9
policy = 0.25
n_rows, n_cols = env.value_function.shape
n_iter = 0

while True:  # Loop until convergence
    delta = 0  # Track changes for convergence
    new_value_function = np.copy(env.value_function)  # Copy to avoid overwriting

    for row in range(n_rows):
        for col in range(n_cols):
            state = (row, col)
            value_sum = 0  # Expected value over actions
            
            for action in agent.actions:
                next_state, reward = env.mdp_dynamics(state, action)
                r, c = next_state
                value_sum += policy * (reward + gamma * env.value_function[r][c])

            new_value_function[row, col] = value_sum
            delta = max(delta, abs(new_value_function[row, col] - env.value_function[row, col]))

    env.value_function = new_value_function  # Update the value function
    n_iter += 1
    if delta < 1e-4:  # Convergence check
        break

print(n_iter)