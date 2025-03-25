import numpy as np

arms = 10
epsilon = 0.1
num_runs = 2000
num_iter = 1000
action_counters = np.zeros(arms)    # for each action, the number of times that the action was chosen
Q_a = np.zeros(arms)                # Action-value estimates
q_star = np.random.normal(size=(arms, num_runs))  # True action values
randomness = np.random.RandomState()              # Random state generator


def setup():
    '''Initializes the true action-value function and sets the seed for each of the num_runs iterations'''
    global Q_a, action_counters, q_star, randomness
    randomness = np.random.RandomState()  # Reset randomness state
    for task in range(num_runs):
        for arm in range(arms):
            q_star[arm, task] = np.random.normal(loc=0.0, scale=1.0)
        randomness.seed(task)  # Setting different seed for each task. A seed enables us to create reproducible streams of random numbers

def reset():
    '''Resets the estimates and the action counter'''
    global Q_a, action_counters
    Q_a.fill(0.0)              # Reset Q_a values
    action_counters.fill(0)    # Reset action counters

def epsilon_greedy_action(epsilon):
    # if np.random.rand() < epsilon: # this is a uniform distribution (incorrect?)
    if np.random.binomial(n=1, p=epsilon):
        return np.random.choice(arms)  # Random action
    else:
        return np.argmax(Q_a)  # Greedy action

def get_reward(a, task_num):
    return q_star[a, task_num] + np.random.normal()  # Reward with noise

def update_estimate(action_idx, reward):
    global Q_a, action_counters
    action_counters[action_idx] += 1
    Q_a[action_idx] += (reward - Q_a[action_idx]) / action_counters[action_idx]  # Eq. 2.3

def arg_max_random_tiebreak(array):
    best_value = array[0]
    best_args = [0]
    for i in range(1, len(array)):
        value = array[i]
        if value < best_value:
            continue
        elif value > best_value:
            best_value = value
            best_args = [i]
        else:
            best_args.append(i)
    return np.random.choice(best_args), best_value

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
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(average_reward, label='Average Reward')
plt.plot(av_oap, label='Average Optimal Action Percentage (AV-OAP)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Average Reward and Optimal Action Percentage')
plt.legend()
plt.grid(True)
plt.show()
