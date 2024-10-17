import numpy as np
from collections import defaultdict 
from models.environment import Environment
from models.policy import Policy
from typing import List, Tuple

class MonteCarloTrainer:
    def __init__(self, env: Environment, policy: Policy, gamma: float = 0.9):
        self.env = env
        self.policy = policy
        self.gamma = gamma

    def generate_episode(self) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int], float]]:
        episode = []
        state = self.env.reset()
        while True:
            action = self.policy.get_action(state)
            next_state, reward, done = self.env.step(state, action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def train(self, num_episodes: int):
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            W = 1
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                self.policy.update(state, action, G, W)
                W /= self.policy.get_action_probability(state, action)
                if W == 0:
                    break
