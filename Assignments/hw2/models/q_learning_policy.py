import numpy as np
from typing import Tuple
from collections import defaultdict
from models.environment import Environment


class DoubleQLearningPolicy:
    def __init__(
        self,
        env: Environment,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        q_init_low: float = 0,
        q_init_high: float = 1,
    ):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_init_low = q_init_low
        self.q_init_high = q_init_high

        self.actions = list(
            range(9)
        )  # Assuming 9 possible actions as in the original code

        # Initialize Q1 and Q2
        self.Q1 = defaultdict(self.initialize_q_values)
        self.Q2 = defaultdict(self.initialize_q_values)

        for finish_pos in self.env.finish_positions:
            self.Q1[(*finish_pos, 0, 0)] = np.zeros(len(self.actions))
            self.Q2[(*finish_pos, 0, 0)] = np.zeros(len(self.actions))

    def initialize_q_values(self):
        return np.random.uniform(self.q_init_low, self.q_init_high, len(self.actions))

    def get_action(self, state: Tuple[int, int, int, int]) -> int:
        if np.random.random() < 0.1:
            return 4 # index of (0,0)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.Q1[state] + self.Q2[state])

    def update(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
        done: bool,
    ):
        if np.random.random() < 0.5:
            self.update_Q1(state, action, reward, next_state, done)
        else:
            self.update_Q2(state, action, reward, next_state, done)

    def update_Q1(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
        done: bool,
    ):
        best_next_action = np.argmax(self.Q1[next_state])
        td_target = reward + (
            0 if done else self.gamma * self.Q2[next_state][best_next_action]
        )
        td_error = td_target - self.Q1[state][action]
        self.Q1[state][action] += self.alpha * td_error

    def update_Q2(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
        done: bool,
    ):
        best_next_action = np.argmax(self.Q2[next_state])
        td_target = reward + (
            0 if done else self.gamma * self.Q1[next_state][best_next_action]
        )
        td_error = td_target - self.Q2[state][action]
        self.Q2[state][action] += self.alpha * td_error

    def train(self, num_episodes: int):
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(state, action)
                self.update(state, action, reward, next_state, done)
                state = next_state

    def get_best_action(self, state: Tuple[int, int, int, int]) -> int:
        return np.argmax(self.Q1[state] + self.Q2[state])
