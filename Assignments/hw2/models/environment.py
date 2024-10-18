import numpy as np
from typing import Tuple, List


class Environment:
    def __init__(
        self,
        track: np.ndarray,
        start_positions: np.ndarray,
        finish_positions: np.ndarray,
    ):
        self.track = track
        self.height, self.width = track.shape
        self.start_positions = start_positions
        self.finish_positions = finish_positions
        self.max_velocity = 5
        self.actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]

    def reset(self) -> Tuple[int, int, int, int]:
        pos = self.start_positions[np.random.randint(len(self.start_positions))]
        return (*pos, 0, 0)  # (x, y, vx, vy)

    def reward(self, is_out_of_bound: bool, is_terminal: bool) -> int:
        if is_out_of_bound:
            return -5
        if is_terminal:
            return 100
        return -1

    def is_terminal(self, new_x: int, new_y: int) -> bool:
        return any(
            (new_x, new_y) == tuple(finish_pos) for finish_pos in self.finish_positions
        )

    def is_out_of_bound(self, new_x: int, new_y: int):
        return (
            new_x < 0
            or new_x >= self.height
            or new_y < 0
            or new_y >= self.width
            or self.track[new_x, new_y] == 1
        )

    def step(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[Tuple[int, int, int, int], float, bool]:
        x, y, vx, vy = state
        ax, ay = self.actions[action]

        # Update velocity
        new_vx = np.clip(vx + ax, -self.max_velocity, self.max_velocity)
        new_vy = np.clip(vy + ay, -self.max_velocity, self.max_velocity)

        # Update position
        new_x, new_y = x + new_vx, y + new_vy

        is_out_of_bound = self.is_out_of_bound(new_x, new_y)
        is_terminal = self.is_terminal(new_x, new_y)

        reward = self.reward(is_out_of_bound, is_terminal)
        next_state = self.reset() if is_out_of_bound else (new_x, new_y, new_vx, new_vy)

        return next_state, reward, is_terminal
