import numpy as np
import random

class GridworldEnvironment():
    def __init__(self):
        self.max_value_function = np.zeros((5, 5))
        self.best_policy_function = np.zeros((5, 5), dtype='<U4') # for each state we can have at most 4 optimal policy values
    
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
        if action == 'n': # north
            next_state = (max(row - 1, 0), col)
            reward = -1 if row == 0 else 0      # Penalize boundary collision
        elif action == 's': # south
            next_state = (min(row + 1, 4), col)
            reward = -1 if row == 4 else 0
        elif action == 'e': # east
            next_state = (row, min(col + 1, 4))
            reward = -1 if col == 4 else 0
        elif action == 'w': # west
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
n_rows, n_cols = env.max_value_function.shape
n_iter = 0

while True:  # Loop until convergence
    delta = 0  # Track changes for convergence
    new_max_value_function = np.copy(env.max_value_function)  # We copy because of Python's Reference-Based Assignment
    new_best_policy_function = np.copy(env.best_policy_function) 

    for row in range(n_rows):
        for col in range(n_cols):
            state = (row, col)
            max_value = 0  # Max value over actions
            best_action = str()
            
            for action in agent.actions:
                next_state, reward = env.mdp_dynamics(state, action)
                r, c = next_state
                if(reward + gamma * env.max_value_function[r][c] >= max_value):
                    max_value = reward + gamma * env.max_value_function[r][c]
                    best_action += action

            new_max_value_function[row, col] = max_value
            new_best_policy_function[row, col] = best_action
            delta = max(delta, abs(new_max_value_function[row, col] - env.max_value_function[row, col]))

    env.max_value_function = new_max_value_function      # Update the value function
    env.best_policy_function = new_best_policy_function  # Update the best policy function

    n_iter += 1
    if delta < 1e-4:  # Convergence check
        break

print(n_iter)