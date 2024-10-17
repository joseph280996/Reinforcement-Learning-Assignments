import numpy as np
from typing import Tuple
from models.environment import Environment

class Policy:
    def __init__(self, env: Environment, epsilon: float = 0.1):
        self.env = env
        self.epsilon = epsilon
        self.slip_probability = 0.1
        
        # Define the state space dimensions
        self.state_dims = (env.height, env.width, 2*env.max_velocity+1, 2*env.max_velocity+1)
        
        # Define the action space
        self.actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]
        self.num_actions = len(self.actions)
        
        # Create Q and C tables using NumPy arrays
        self.Q = np.zeros(self.state_dims + (self.num_actions,))
        self.C = np.zeros(self.state_dims + (self.num_actions,))

    def state_to_index(self, state: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x, y, vx, vy = state
        return (x, y, vx + self.env.max_velocity, vy + self.env.max_velocity)

    def get_action(self, state: Tuple[int, int, int, int]) -> Tuple[int, int]:
        if np.random.random() < self.slip_probability:
            return (0, 0)
        elif np.random.random() < self.epsilon:
            return self.actions[np.random.randint(self.num_actions)]
        else:
            state_index = self.state_to_index(state)
            action_index = np.argmax(self.Q[state_index])
            return self.actions[action_index]

    def update(self, states: np.ndarray, actions: np.ndarray, G: np.ndarray, W: np.ndarray):
        for state, action, g, w in zip(states, actions, G, W):
            state_index = self.state_to_index(tuple(state))
            action_index = self.actions.index(tuple(action))
            self.C[state_index + (action_index,)] += w
            self.Q[state_index + (action_index,)] += (w / self.C[state_index + (action_index,)]) * (g - self.Q[state_index + (action_index,)])

    def get_action_probability(self, state: Tuple[int, int, int, int], action: Tuple[int, int]) -> float:
        state_index = self.state_to_index(state)
        best_action_index = np.argmax(self.Q[state_index])
        best_action = self.actions[best_action_index]

        if action == (0, 0):
            return self.slip_probability + (1 - self.slip_probability) * (self.epsilon / self.num_actions)
        elif action == best_action:
            return (1 - self.slip_probability) * (1 - self.epsilon + self.epsilon / self.num_actions)
        else:
            return (1 - self.slip_probability) * (self.epsilon / self.num_actions)