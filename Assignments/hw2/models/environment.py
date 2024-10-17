import numpy as np
from typing import Set, Tuple

class Environment:
    def __init__(self, track: np.ndarray, start_positions: Set[Tuple[int, int]], finish_line: Set[Tuple[int, int]]):
        self.track = track
        self.height, self.width = track.shape
        self.start_positions = list(start_positions)
        self.finish_line = finish_line
        self.max_velocity = 5
        self.slip_probability = 0.1

    def reset(self) -> Tuple[int, int, int, int]:
        pos = self.start_positions[np.random.randint(len(self.start_positions))]
        return (*pos, 0, 0)  # (x, y, vx, vy)
    
    def reward(self, state, action, next_state) -> int:
        return 100 if self.is_terminal(next_state) else -1

    def is_terminal(self, next_state: Tuple[int, int]) -> bool:
        return next_state in self.finish_line

    def is_out_of_bound(self, next_state: Tuple[int, int]):
        new_x, new_y, _, _ = next_state
        return new_x < 0 or new_x >= self.height or new_y < 0 or new_y >= self.width or self.track[new_x, new_y] == 0

    def step(self, state: Tuple[int, int, int, int], action: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], float, bool]:
        x, y, vx, vy = state
        ax, ay = action

        # Apply the stochastic action constraint
        if np.random.random() < self.slip_probability:
            ax, ay = 0, 0

        # Update velocity
        new_vx = max(-self.max_velocity, min(self.max_velocity, vx + ax))
        new_vy = max(-self.max_velocity, min(self.max_velocity, vy + ay))

        # Update position
        new_x, new_y = x + new_vx, y + new_vy
        next_state = (new_x, new_y, new_vx, new_vy)
        is_out_of_bound = self.is_out_of_bound(next_state)

        reward = self.reward(state, action, next_state)

        if is_out_of_bound: 
            next_state = self.reset()

        return next_state, reward, self.is_terminal(next_state)  # Continue, -1 reward for each step
