import numpy as np

class RacetrackEnvironment:
    def __init__(self, track):
        self.track = track
        self.start_line, self.finish_line = self._get_start_finish_lines()
        self.max_velocity = 2
        self.actions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

    def _get_start_finish_lines(self):
        start_line = [(i, j) for i in range(self.track.shape[0]) for j in range(self.track.shape[1]) if self.track[i, j] == 3]
        finish_line = [(i, j) for i in range(self.track.shape[0]) for j in range(self.track.shape[1]) if self.track[i, j] == 2]
        return start_line, finish_line

    def is_valid_position(self, pos):
        return 0 <= pos[0] < self.track.shape[0] and 0 <= pos[1] < self.track.shape[1] and self.track[pos] != 1

    def get_next_position(self, pos, vel):
        return (pos[0] + vel[0], pos[1] + vel[1])

    def check_collision(self, start_pos, end_pos):
        x0, y0 = start_pos
        x1, y1 = end_pos
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if not self.is_valid_position((x, y)):
                return True
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return False

    def reward_function(self, current_state, action, next_state):
        if self.is_terminal_state(next_state):
            return 100
        else:
            return -1

    def step(self, current_state, action):
        pos, vel = current_state
        
        # Update velocity based on action
        new_vel = (
            max(-self.max_velocity, min(self.max_velocity, vel[0] + action[0])),
            max(-self.max_velocity, min(self.max_velocity, vel[1] + action[1]))
        )
        
        # Ensure velocity is not zero unless at start line
        if new_vel == (0, 0) and pos not in self.start_line:
            new_vel = vel
        
        new_pos = self.get_next_position(pos, new_vel)

        if self.check_collision(pos, new_pos):
            return (self.start_line[np.random.randint(len(self.start_line))], (0, 0))
        else:
            return (new_pos, new_vel)

    def is_terminal_state(self, state):
        return state[0] in self.finish_line

    def take_action(self, state, action):
        next_state = self.step(state, action)
        reward = self.reward_function(state, action, next_state)
        done = self.is_terminal_state(next_state)
        return next_state, reward, done

    def get_valid_actions(self, state):
        _, vel = state
        return [action for action in self.actions if 
                -self.max_velocity <= vel[0] + action[0] <= self.max_velocity and
                -self.max_velocity <= vel[1] + action[1] <= self.max_velocity]

