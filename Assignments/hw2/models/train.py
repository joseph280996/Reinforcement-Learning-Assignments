import numpy as np
from collections import defaultdict 

class MonteCarloTrainer:
    def __init__(self, environment, policy, learning_rate, discount_factor):
        self.environment = environment
        self.policy = policy
        self.returns = defaultdict(lambda: defaultdict(list))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def train_step(self, state, action, reward, next_state, next_action):
        current_q = self.policy.get_q_value(state, action)
        next_q = self.policy.get_q_value(next_state, next_action)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.policy.update(state, action, new_q)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = (self.environment.start_line[np.random.randint(len(self.environment.start_line))], (0, 0))
            action = self.policy.get_action(state)
            
            while True:
                next_state, reward, done = self.environment.take_action(state, action)
                next_action = self.policy.get_action(next_state)
                
                self.train_step(state, action, reward, next_state, next_action)
                
                if done:
                    break
                
                state = next_state
                action = next_action

            if episode % 1000 == 0:
                print(f"Episode {episode} completed")

    def get_optimal_path(self):
        state = (self.environment.start_line[0], (0, 0))
        path = [state[0]]
        
        while not self.environment.is_terminal_state(state):
            action = self.policy.get_action(state)
            state, _, _ = self.environment.take_action(state, action)
            path.append(state[0])
        
        return path

    def train(self, num_episodes, gamma=1.0, epsilon=0.1):
        for _ in range(num_episodes):
            episode = self._generate_episode(epsilon)
            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = gamma * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                    self.returns[state][action].append(G)
                    self.policy.update(state, action, np.mean(self.returns[state][action]))

    def _generate_episode(self, epsilon):
        episode = []
        state = (self.environment.start_line[np.random.randint(len(self.environment.start_line))], (0, 0))
        done = False

        while not done:
            action = self.policy.get_action(state, epsilon)
            if np.random.random() < 0.1:
                action = (0, 0)
            next_state, reward, done = self.environment.take_action(state, action)
            episode.append((state, action, reward))
            state = next_state

        return episode
