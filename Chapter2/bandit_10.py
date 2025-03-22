import numpy as np
import matplotlib.pyplot as plt

def plot():
    # Plot average rewards and optimal action percentage
    plt.figure(figsize=(12, 5))

    # Plot reward over time
    plt.subplot(1, 2, 1)
    plt.plot(average_rewards, label="Average Reward")
    # plt.plot(rewards, label="Reward") # very noisy
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward")
    plt.title("Reward vs. Time Steps")
    plt.legend()

    # Plot optimal action selection percentage
    plt.subplot(1, 2, 2)
    plt.plot(optimal_action_percentage, label="Optimal Action %", color='orange')
    plt.xlabel("Time Steps")
    plt.ylabel("Percentage")
    plt.title("Optimal Action Selection %")
    plt.legend()

    # Plot the evolution of Q_a error over time
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab10", arms)

    for i in range(arms):
        plt.plot(Q_a_error_history[:, i], label=f"Error[{i}]", color=colors(i))

    plt.xlabel("Time Steps")
    plt.ylabel("Absolute Error |Q_a - q*|")
    plt.title("Convergence of Action-Value Estimates (Error over Time)")
    plt.legend()


    plt.show()

class BanditEnvironment:
    def __init__(self, arms=10):
        self.q_star = np.random.normal(loc=0, scale=1, size=arms) # mean, variance, shape of q_star
        self.optimal_action = np.argmax(self.q_star)  # Best action

    def get_reward(self, action_idx):
        return np.random.normal(loc=self.q_star[action_idx], scale=1) # we get one reward for choosing the action indexed by action_idx

class EpsilonGreedyAgent:
    def __init__(self, arms=10):
        self.Q_a = np.zeros(shape=(arms,))                 # action-value estimates
        self.action_counters = np.zeros(shape=(arms,))     # for each action, the number of times that the action was chosen 
        self.cumulative_rewards = np.zeros(shape=(arms,))  # numerator of Eq. 2.1
        self.epsilon = 0.1

    def greedy_action(self):
        return np.argmax(self.Q_a)
    
    def epsilon_greedy_action(self):
        '''
        Choose the greedy (i.e., the best) action with probability 1 - self.epsilon. For that,
        we need a Bernoulli random variable, with parameter self.epsilon, which is a special
        case of the Binomial random variable, with a number of trials equal to 1. p = prob(1).
        0 => choose greedy_action, 1 => choose epsilon_greedy_action
        '''
        if np.random.binomial(n=1, p=self.epsilon):
            random_action_idx = np.random.choice(len(self.Q_a))
            return random_action_idx
        else:
            return self.greedy_action()
    
    def update_estimate(self, reward, action_idx):
        # update the entry of Q_a indexed by action_idx
        self.action_counters[action_idx] = self.action_counters[action_idx] + 1
        self.cumulative_rewards[action_idx] = self.cumulative_rewards[action_idx] + reward
        self.Q_a[action_idx] = 1/self.action_counters[action_idx] * self.cumulative_rewards[action_idx]

arms  = 10
env   = BanditEnvironment(arms)
agent = EpsilonGreedyAgent(arms)

numIter = 1000
rewards = []
optimal_action_counts = []
Q_a_error_history = np.zeros((numIter, arms))

for t in range(numIter):
    action_idx = agent.epsilon_greedy_action()
    reward = env.get_reward(action_idx)
    agent.update_estimate(reward, action_idx)

    rewards.append(reward) # Store rewards for tracking average performance
    optimal_action_counts.append(action_idx == env.optimal_action) # number of times the optimal action in env.q_star was chosen
    Q_a_error_history[t] = np.abs(agent.Q_a - env.q_star)  # Compute and store absolute error between q_star and Q_a


# Filtering: Compute average rewards and optimal action percentage (for smooth plots)
average_rewards = np.cumsum(rewards) / (np.arange(numIter) + 1)
optimal_action_percentage = np.cumsum(optimal_action_counts) / (np.arange(numIter) + 1) * 100

plot()