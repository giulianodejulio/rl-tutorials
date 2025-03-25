import numpy as np
import matplotlib.pyplot as plt

arms = 10
epsilon = 0.1
num_runs = 2000
num_iter = 1000
action_counters = np.zeros(arms)        # Number of times each action was chosen
Q_a = np.zeros(arms)                    # Action-value estimates
randomness = np.random.RandomState(42)  # Fixed seed

def setup(seed=42):
    '''Initializes the true action-value function and sets the seed for reproducibility'''
    global Q_a, action_counters, q_star, randomness
    randomness = np.random.RandomState(seed)  # Fixed seed
    q_star = randomness.normal(loc=0.0, scale=1.0, size=(arms, num_runs))  # Fixed action values

def reset():
    '''Resets the estimates and the action counter'''
    global Q_a, action_counters
    Q_a.fill(0.0)              # Reset Q_a values
    action_counters.fill(0)    # Reset action counters

def epsilon_greedy_action(epsilon):
    if randomness.binomial(n=1, p=epsilon):
        return randomness.choice(arms)  # Random action
    else:
        return np.argmax(Q_a)  # Greedy action

def get_reward(action_idx, run):
    return q_star[action_idx, run] + randomness.normal()

def update_estimate(action_idx, reward):
    global Q_a, action_counters
    action_counters[action_idx] += 1
    Q_a[action_idx] += (reward - Q_a[action_idx]) / action_counters[action_idx]  # Eq. 2.3

def runs(num_runs=num_runs, num_iter=num_iter, epsilon=0):
    average_reward = np.zeros(num_iter)
    av_oap = np.zeros(num_iter)
    
    for run in range(num_runs):
        optimal_action = np.argmax(q_star[:, run])
        reset()
        
        for t in range(num_iter):
            action_idx = epsilon_greedy_action(epsilon)
            reward = get_reward(action_idx, run)
            update_estimate(action_idx, reward)
            
            average_reward[t] += reward
            if action_idx == optimal_action:
                av_oap[t] += 1
    
    # Normalize the results
    average_reward /= num_runs
    av_oap /= num_runs
    
    return average_reward, av_oap


setup()
average_reward, av_oap = runs(num_runs=num_runs, num_iter=num_iter, epsilon=0.1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(average_reward, label='Average Reward')
plt.plot(av_oap, label='Average Optimal Action Percentage (AV-OAP)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Average Reward and Optimal Action Percentage')
plt.legend()
plt.grid(True)
plt.show()
