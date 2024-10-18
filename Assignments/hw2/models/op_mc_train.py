import numpy as np
from typing import Tuple, List

from models.environment import Environment
from models.policy import Policy
from models.visualize import visualize_episode, visualize_episode_with_velocity


def get_start_finish_lines(
    track: np.ndarray,
) -> Tuple[int, np.ndarray, int, np.ndarray]:
    start_positions = np.argwhere(track == 3)

    finish_positions = np.argwhere(track == 2)

    return start_positions, finish_positions


class MonteCarloTrainer:
    def __init__(self, track: np.ndarray, gamma: float = 0.9):
        start_positions, finish_positions = get_start_finish_lines(track)
        self.env = Environment(track, start_positions, finish_positions)
        self.policy = Policy(self.env)
        self.gamma = gamma
        self.ep_rewards = []

    def generate_episode(
        self,
    ) -> Tuple[
        List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], int, int
    ]:
        state = self.env.reset()
        episode = []
        ep_reward = 0
        done = False

        while not done:
            action = self.policy.behavior_policy(state)

            next_state, reward, done = self.env.step(state, action)

            episode.append((state, action, reward))

            ep_reward += reward

            state = next_state

        self.ep_rewards.append(ep_reward)

        return episode

    def train(self, num_episodes: int) -> None:
        for _ in range(num_episodes):
            episode = self.generate_episode()

            G = 0
            W = 1
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                self.policy.update(state, action, G, W)
                pi = self.policy.get_best_action(state)
                if action != pi:
                    break

                W /= self.policy.get_action_probs(state)[action]

    def train_with_visual(self, num_episodes: int) -> None:
        state = self.env.reset()
        episode = []
        path = [state]
        step_count = 0
        ep_rewards = 0
        while True:
            action = self.policy.behavior_policy(state)
            next_state, reward, done = self.env.step(state, action)

            episode.append((state, action, reward))
            path.append(next_state)
            ep_rewards += reward
            step_count += 1

            if done:
                break
            state = next_state

        print(f"Total Steps taken: {step_count}")

        # Visualize the first episode
        visualize_episode(self.env.track, path)
        visualize_episode_with_velocity(self.env.track, path)

        G = 0
        W = 1
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            self.policy.update(state, action, G, W)
            pi = self.policy.get_best_action(state)
            if action != pi:
                break

            W /= self.policy.get_action_probs(state)[action]
