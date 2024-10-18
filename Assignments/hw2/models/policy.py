import numpy as np
from collections import defaultdict
from typing import Tuple
from models.environment import Environment

class Policy:
    def __init__(self, env: Environment, epsilon: float = 0.1, q_init_low: float = 0, q_init_high: float = 1):
        self.env = env
        self.actions = list(range(9))
        self.Q = defaultdict(lambda: np.random.uniform(q_init_low, q_init_high, len(self.actions)))
        self.C = defaultdict(lambda: np.zeros(len(self.actions)))
        self.epsilon = epsilon
        self.slipped_prob = 0.1
        self.zero_zero_idx = self.env.actions.index((0,0))

    def target_policy(self, state: Tuple[int, int, int, int]) -> Tuple[int, int]:
        return self.get_best_action(state)

    def get_best_action(self, state: Tuple[int, int, int, int]) -> Tuple[int, int]:
        q_values = self.Q[state]
        if np.any(q_values == 0.0):
            return np.random.choice(self.actions)
        return np.argmax(q_values)

    def update(self, state: Tuple[int, int, int, int], action: int, G: float, W: float):
        self.C[state][action] += W
        self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
    
    def behavior_policy(self, state: Tuple[int, int, int, int]) -> Tuple[int, int]:
        probabilities = self.get_action_probs(state)
        return np.random.choice(self.actions, p=probabilities)

    def get_action_probs(self, state: Tuple[int, int, int, int]) -> float:
        best_action = self.get_best_action(state)
        n_actions = len(self.actions)
        
        # Initialize probabilities
        probs = np.full(n_actions, self.epsilon / n_actions)
        probs[best_action] += (1 - self.epsilon)
        
        # Add probability for (0,0) action
        probs = probs * (1 - self.slipped_prob)
        probs[self.zero_zero_idx] += self.slipped_prob

        return probs