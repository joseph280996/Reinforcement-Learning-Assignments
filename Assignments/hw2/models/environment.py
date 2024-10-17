import numpy as np
from typing import Tuple, List

class Environment:
    def __init__(self, track: np.ndarray, start_line: int, start_positions: List[int], finish_line: int, finish_positions: List[int]):
        self.track = track
        self.height, self.width = track.shape
        self.start_line = start_line
        self.start_positions = np.array([(start_line, y) for y in start_positions])
        self.finish_line = set((finish_line, y) for y in finish_positions)
        self.max_velocity = 5

    def reset(self) -> np.ndarray:
        pos = self.start_positions[np.random.randint(len(self.start_positions))]
        return np.array([*pos, 0, 0])  # (x, y, vx, vy)
    
    def reward(self, is_out_of_bound: bool, is_terminal: bool) -> int:
        if is_out_of_bound: return -5
        if is_terminal: return 100
        return -1

    def is_terminal(self, new_x: int, new_y: int) -> bool:
        return (new_x, new_y) in self.finish_line

    def is_out_of_bound(self, new_x: int, new_y: int):
        return new_x < 0 or new_x >= self.height or new_y < 0 or new_y >= self.width or self.track[new_x, new_y] == 1
    
    def step(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        x, y, vx, vy = state
        ax, ay = action

        # Update velocity
        new_vx = np.clip(vx + ax, -self.max_velocity, self.max_velocity)
        new_vy = np.clip(vy + ay, -self.max_velocity, self.max_velocity)

        # Update position
        new_x, new_y = x + new_vx, y + new_vy

        is_out_of_bound = self.is_out_of_bound(new_x, new_y)
        is_terminal = self.is_terminal(new_x, new_y)

        reward = self.reward(is_out_of_bound, is_terminal)
        next_state = self.reset() if is_out_of_bound else np.array([new_x, new_y, new_vx, new_vy])

        return next_state, reward, is_terminal