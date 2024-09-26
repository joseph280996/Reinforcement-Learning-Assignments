import numpy as np

class Bandit():
    def __init__(self, n_steps, alpha=None, use_gradient = False, seed = 42):
        self.alpha = alpha
        self.use_gradient = use_gradient

        self.n_actions = 10 # Because it's 10-armed so we have 10 possible actions to take at each step. 
        self.n_steps = n_steps
        self.n_runs = 2000
        self.epsilon = 0.1
        self.std_walk = 0.01
        self.seed = 42

        self.rewards_at_step = np.zeros(self.n_steps)
        self.optimal_reward_at_step = np.zeros(self.n_steps)
        self.H = np.zeros(self.n_steps)

    def run(self):
        np.random.seed(self.seed)

        for run in range(self.n_runs):
            q_true = np.zeros(self.n_actions)  # True action values
            q_estimates = np.zeros(self.n_actions)  # Estimated action values
            action_counts = np.zeros(self.n_actions)
            self.avg_reward = 0

            for step in range(self.n_steps):

                # Action selection
                action = self.__action_selection(q_estimates)

                # Receive reward
                reward = np.random.normal(q_true[action], 1)
                self.rewards_at_step[step] += reward

                action_counts[action] += 1

                alpha = self.alpha if self.alpha is not None else 1 / action_counts[action]

                if self.use_gradient:
                    self.__update_preference(action, alpha, reward)
                else:
                    # Update estimate using sample average
                    q_estimates[action] += alpha * (reward - q_estimates[action])

                # Update true action values (random walk)
                q_true += np.random.normal(0, self.std_walk, self.n_actions)
                optimal_action = np.argmax(q_true)

                # Logging results
                self.optimal_reward_at_step[step] += (1 if action == optimal_action else 0)

        return self.rewards_at_step / self.n_runs, self.optimal_reward_at_step / self.n_runs

    def __action_selection(self, q_estimates):
        if self.use_gradient:
            action_probs = self.__softmax()
            return np.random.choice(self.n_actions, p=action_probs)

        if np.random.rand() < self.epsilon: 
            return np.random.choice(self.n_actions) 

        return np.argmax(q_estimates)

    def __update_preferences(self, action, alpha, current_reward):
        action_probs = self.__softmax()
        baseline = self.avg_reward
        for a in range(self.n_actions):
            if a == action:
                self.H[a] += alpha * (current_reward - baseline) * (1 - action_probs[a])
            else:
                self.H[a] -= alpha * (current_reward - baseline) * action_probs[a]

        self.avg_reward += (current_reward - self.avg_reward) / (self.step + 1)

    def __softmax(self):
        exp_prefs = np.exp(self.H - np.max(self.H)) # for numerical stability
        return exp_prefs/np.sum(exp_prefs)

