# Nonstationary problem
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
        plt.plot(Q_a_error_history[:, i], label=f"Error[{i+1}]", color=colors(i))

    plt.xlabel("Time Steps")
    plt.ylabel("Absolute Error |Q_a - q*|")
    plt.title("Convergence of Action-Value Estimates (Error over Time)")
    plt.legend()
    plt.show()


class BanditEnvironment:
    def __init__(self, arms=10):
        self.q_star = np.zeros(shape=(arms,))
        self.optimal_action = np.argmax(self.q_star)

    def get_reward(self, action_idx):
        return np.random.normal(loc=self.q_star[action_idx], scale=1)

class EpsilonGreedyAgent:
    def __init__(self, arms=10, epsilon=0.1):
        self.Q_a = np.zeros(shape=(arms,))
        self.epsilon = epsilon

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

    def update_estimate_2(self, reward, action_idx, iter):
        # alpha = 1/(iter+1)
        alpha = 0.1
        self.Q_a[action_idx] = self.Q_a[action_idx] + alpha * (reward - self.Q_a[action_idx])


arms    = 10
epsilon = 0.1
env   = BanditEnvironment(arms)
agent = EpsilonGreedyAgent(arms, epsilon)

numIter = int(1e4)
rewards = []
optimal_action_counts = []
Q_a_error_history = np.zeros((numIter, arms))

for t in range(numIter):
    env.q_star = env.q_star + np.random.normal(loc=0, scale=0.01, size=arms)
    action_idx = agent.epsilon_greedy_action()
    reward = env.get_reward(action_idx)
    agent.update_estimate_2(reward, action_idx, t)

    rewards.append(reward)
    optimal_action_counts.append(action_idx == env.optimal_action)
    Q_a_error_history[t] = np.abs(agent.Q_a - env.q_star)


average_rewards = np.cumsum(rewards) / (np.arange(numIter) + 1)
optimal_action_percentage = np.cumsum(optimal_action_counts) / (np.arange(numIter) + 1) * 100

print(f"The best action in the environment is {env.optimal_action + 1}")
plot()