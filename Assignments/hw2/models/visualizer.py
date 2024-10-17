import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple
import numpy as np

class Visualizer:
    def __init__(self, env):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, env.width)
        self.ax.set_ylim(0, env.height)
        self.ax.invert_yaxis()

    def plot_track(self):
        self.ax.imshow(self.env.track, cmap='binary')
        for pos in self.env.start_positions:
            self.ax.add_patch(Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1, fill=False, edgecolor='g', lw=2))
        for pos in self.env.finish_line:
            self.ax.add_patch(Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1, fill=False, edgecolor='r', lw=2))

    def plot_trajectory(self, trajectory: List[Tuple[int, int]], color='b', alpha=0.5):
        x, y = zip(*trajectory)
        self.ax.plot(y, x, color=color, linewidth=2, alpha=alpha)
        self.ax.plot(y[0], x[0], 'go', markersize=10)  # Start
        self.ax.plot(y[-1], x[-1], 'ro', markersize=10)  # End

    def plot_multiple_trajectories(self, trajectories: List[List[Tuple[int, int]]], num_trajectories=5):
        colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
        for trajectory, color in zip(trajectories[:num_trajectories], colors):
            self.plot_trajectory(trajectory, color=color)

    def plot_episodes(self, trajectories: List[List[Tuple[int, int]]], interval: int = 1000, num_episodes: int = 5):
        """Plot trajectories from different stages of training."""
        self.plot_track()
        episodes_to_plot = list(range(0, len(trajectories), interval))[-num_episodes:]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(episodes_to_plot)))
        for episode, color in zip(episodes_to_plot, colors):
            self.plot_trajectory(trajectories[episode], color=color, alpha=0.7)
            self.ax.text(0.5, 0.95, f'Episodes: {episodes_to_plot}', transform=self.ax.transAxes, ha='center')

    def show(self):
        plt.show()

    def clear_trajectories(self):
        for line in self.ax.lines:
            line.remove()