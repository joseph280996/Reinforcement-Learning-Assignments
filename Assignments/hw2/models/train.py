import numpy as np
from models.environment import Environment
from models.policy import Policy
from typing import Tuple

def get_start_finish_lines(track):
    start_line = np.where(np.any(track == 3, axis=1))[0][0]
    start_positions = np.where(track[start_line] == 3)[0]
    
    finish_line = np.where(np.any(track == 2, axis=1))[0][0]
    finish_positions = np.where(track[finish_line] == 2)[0]
    
    return start_line, start_positions, finish_line, finish_positions

class MonteCarloTrainer:
    def __init__(self, track: np.ndarray, gamma: float = 0.9):
        start_line, start_positions, finish_line, finish_positions = get_start_finish_lines(track)
        self.env = Environment(track, start_line, start_positions, finish_line, finish_positions)
        self.policy = Policy(self.env)
        self.gamma = gamma
        self.total_returns = 0

    def generate_episode(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states, actions, rewards = [], [], []
        state = self.env.reset()
        step_count = 0
        while True:
            action = self.policy.get_action(state)
            next_state, reward, done = self.env.step(state, action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            step_count += 1

            if done:
                break
            state = next_state
        
        print(f"Total Steps taken: {step_count}")
        
        # Convert lists to NumPy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        # Calculate G and W for each step
        G = np.zeros(len(rewards))
        W = np.ones(len(rewards))
        G[-1] = rewards[-1]
        for t in range(len(rewards) - 2, -1, -1):
            G[t] = rewards[t] + self.gamma * G[t + 1]
            W[t] = W[t + 1] / self.policy.get_action_probability(tuple(states[t]), tuple(actions[t]))
        
        return states, actions, G, W

    def train(self, num_episodes: int):
        for ep_id in range(num_episodes):
            print(f"\n---- Episode: {ep_id} ----\n")
            states, actions, G, W = self.generate_episode()
            
            # Update Q-values and C-values using vectorized operations
            self.policy.update(states, actions, G, W)
            
            self.total_returns += G[0]  # G[0] is the total return for the episode

        print(f"Average return over {num_episodes} episodes: {self.total_returns / num_episodes}")