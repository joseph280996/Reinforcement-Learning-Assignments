import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_episode(track, episode_path):
    # Create a color map
    cmap = ListedColormap(["white", "black", "green", "blue", "red"])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(track, cmap=cmap)

    # Plot the path
    path = np.array(episode_path)
    ax.plot(path[:, 1], path[:, 0], color="red", linewidth=2, marker="o", markersize=4)

    # Add start and end points
    ax.plot(
        path[0, 1], path[0, 0], color="blue", marker="o", markersize=10, label="Start"
    )
    ax.plot(
        path[-1, 1], path[-1, 0], color="green", marker="o", markersize=10, label="End"
    )

    # Customize the plot
    ax.set_title("Racetrack Episode Visualization")
    ax.legend()
    ax.grid(True)

    plt.show()


def visualize_episode_with_velocity(track, episode_path):
    # Create a color map
    cmap = ListedColormap(["white", "black", "green", "blue", "red"])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(track, cmap=cmap)

    # Plot the path
    path = np.array(episode_path)
    ax.plot(path[:, 1], path[:, 0], color="red", linewidth=2, marker="o", markersize=4)

    # Add velocity arrows
    for i in range(len(path) - 1):
        ax.arrow(
            path[i, 1],
            path[i, 0],
            path[i, 3],
            path[i, 2],
            color="yellow",
            width=0.1,
            head_width=0.5,
        )

    # Add start and end points
    ax.plot(
        path[0, 1], path[0, 0], color="blue", marker="o", markersize=10, label="Start"
    )
    ax.plot(
        path[-1, 1], path[-1, 0], color="green", marker="o", markersize=10, label="End"
    )

    # Customize the plot
    ax.set_title("Racetrack Episode Visualization with Velocity")
    ax.legend()
    ax.grid(True)

    plt.show()
