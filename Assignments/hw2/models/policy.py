import numpy as np
from collections import defaultdict
from typing import Tuple

class Policy:
    def __init__(self, epsilon: float = 0.1):
        self.Q = {}
        self.C = {}
        self.epsilon = epsilon
        self.actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]
        self.slip_probability = 0.1

    def get_action(self, state: Tuple[int, int, int, int]) -> Tuple[int, int]:
        if np.random.random() < self.epsilon:
            return self.actions[np.random.randint(len(self.actions))]
        else:
            return max(self.actions, key=lambda a: self.Q.get((state, a), 0))

    def update(self, state: Tuple[int, int, int, int], action: Tuple[int, int], G: float, W: float):
        sa_pair = (state, action)
        if sa_pair not in self.C:
            self.C[sa_pair] = 0
            self.Q[sa_pair] = 0
        self.C[sa_pair] += W
        self.Q[sa_pair] += (W / self.C[sa_pair]) * (G - self.Q[sa_pair])

    def get_action_probability(self, state: Tuple[int, int, int, int], action: Tuple[int, int]) -> float:
        best_action = self.get_action(state)
        if action == (0, 0):
            return self.slip_probability + (1 - self.slip_probability) * (
                self.epsilon / len(self.actions) if action != best_action else 1 - self.epsilon + self.epsilon / len(self.actions)
            )
        else:
            return (1 - self.slip_probability) * (
                self.epsilon / len(self.actions) if action != best_action else 1 - self.epsilon + self.epsilon / len(self.actions)
            )
