import numpy as np
from typing import Tuple
from models.environment import Environment
from models.q_learning_policy import DoubleQLearningPolicy
from models.visualize import visualize_episode, visualize_episode_with_velocity


def get_start_finish_lines(track: np.ndarray) -> Tuple[int, np.ndarray, int, np.ndarray]:
    start_positions = np.argwhere(track == 3)
    
    finish_positions = np.argwhere(track==2)
    
    return start_positions, finish_positions

class DoubleQLearningAgent:
    def __init__(self, track: np.ndarray, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.9):
        start_positions, finish_positions = get_start_finish_lines(track)
        self.env = Environment(track, start_positions, finish_positions)
        self.policy = DoubleQLearningPolicy(self.env)
        self.ep_rewards = []

    def train(self, num_episodes: int):
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.policy.get_action(state)
                next_state, reward, done = self.env.step(state, action)

                self.policy.update(state, action, reward, next_state, done)

                ep_reward += reward
                state = next_state
            self.ep_rewards.append(ep_reward)

    def train_with_visual(self, num_episodes: int) -> None:
        state = self.env.reset()
        path = [state]
        step_count = 0
        ep_rewards = 0
        done = False

        while not done:
            action = self.policy.get_action(state)
            next_state, reward, done = self.env.step(state, action)

            self.policy.update(state, action, reward, next_state, done)

            path.append(next_state)

            ep_rewards += reward
            state = next_state
            step_count += 1

        
        print(f"Total Steps taken: {step_count}")
        print(f"Total rewards: {ep_rewards}")

        # Visualize the first episode
        visualize_episode(self.env.track, path)
        visualize_episode_with_velocity(self.env.track, path)