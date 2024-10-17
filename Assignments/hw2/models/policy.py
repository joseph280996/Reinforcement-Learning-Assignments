import numpy as np
from collections import defaultdict

class Policy:
    def __init__(self, environment, epsilon=0.1):
        self.environment = environment
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.environment.get_valid_actions(state)[np.random.randint(0, len(self.environment.get_valid_actions(state)))]
        else:
            return max(self.environment.get_valid_actions(state), key=lambda action: self.q_table[state][action])

    def update(self, state, action, new_q_value):
        self.q_table[state][action] = new_q_value

    def get_q_value(self, state, action):
        return self.q_table[state][action]