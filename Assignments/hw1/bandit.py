import numpy as np

class NonStationaryBandit:
    def __init__(self, k=10):
        self.k = k
        self.q_star = np.zeros(k)
        
    def pull(self, arm):
        return np.random.normal(self.q_star[arm], 1)
    
    def update(self):
        self.q_star += np.random.normal(0, 0.01, self.k)
    
    def optimal_action(self):
        return np.argmax(self.q_star)

def epsilon_greedy(Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(len(Q))
    else:
        return np.argmax(Q)

def softmax(H, temperature=1.0):
    exp_H = np.exp(H / temperature - np.max(H / temperature))
    return exp_H / np.sum(exp_H)

def ucb_select(Q, N, t, c):
    ucb_values = Q + c * np.sqrt(np.log(t + 1) / (N + 1e-5))
    return np.argmax(ucb_values)

def run_experiment(num_steps, algorithm, alpha, epsilon, gradient, baseline, ucb_c=None):
    bandit = NonStationaryBandit()
    k = bandit.k
    
    if gradient:
        H = np.zeros(k)
        pi = softmax(H)
    else:
        Q = np.zeros(k)
        N = np.zeros(k)
    
    rewards = np.zeros(num_steps)
    optimal_actions = np.zeros(num_steps, dtype=int)
    chosen_actions = np.zeros(num_steps, dtype=int)
    
    for t in range(num_steps):
        optimal_action = bandit.optimal_action()
        optimal_actions[t] = optimal_action
        
        if algorithm == 'ucb':
            action = ucb_select(Q, N, t, ucb_c)
        elif gradient:
            action = np.random.choice(k, p=pi)
        else:
            action = epsilon_greedy(Q, epsilon)
        
        chosen_actions[t] = action
        reward = bandit.pull(action)
        rewards[t] = reward
        
        if gradient:
            average_reward = np.mean(rewards[:t+1]) if baseline else 0
            one_hot = np.zeros(k)
            one_hot[action] = 1
            H += alpha * (reward - average_reward) * (one_hot - pi)
            pi = softmax(H)
        elif algorithm == 'sample_average':
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
        else:  # constant step-size and UCB
            N[action] += 1
            Q[action] += alpha * (reward - Q[action])
        
        bandit.update()
    
    return rewards, optimal_actions, chosen_actions