import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from pathlib import Path
import typer
from enum import Enum
from typing import Optional
from datetime import datetime


def calculate_heading(p1, p2):
    """
    Calculate heading angle (theta) between two points in the xy-plane

    Args:
        p1: First point [x, y, z]
        p2: Second point [x, y, z]

    Returns:
        theta: Heading angle in radians
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Default heading if points are too close
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0

    return np.arctan2(dy, dx) - np.pi / 2


def generate_spiral_trajectory(num_points=100, radius=5.0, depth=9.0, num_loops=3):
    """
    Generate a 3D spiral trajectory

    Args:
        num_points: Number of waypoints to generate
        radius: Maximum radius of the spiral
        depth: Maximum depth of the spiral (positive value, applied as negative z, default 9.0m)
        num_loops: Number of complete loops in the spiral

    Returns:
        numpy array of shape (num_points, 3) containing x, y, z coordinates
    """
    t = np.linspace(0, num_loops * 2 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(0, -depth, num_points)

    headings = np.zeros(num_points)
    for i in range(num_points - 1):
        p1 = np.array([x[i], y[i], z[i]])
        p2 = np.array([x[i + 1], y[i + 1], z[i + 1]])
        headings[i] = calculate_heading(p1, p2)
    headings[-1] = headings[-2]  # Last heading same as second last

    return np.column_stack((x, y, z, headings))


def generate_lemniscate_trajectory(num_points=100, scale=5.0, depth_range=(-9.0, 0.0)):
    """
    Generate a 3D lemniscate (figure-eight) trajectory

    Args:
        num_points: Number of waypoints to generate
        scale: Scale factor for x and y coordinates
        depth_range: Tuple of (max_depth, min_depth) for z oscillation (negative values for depth, default -9.0 to 0)

    Returns:
        numpy array of shape (num_points, 3) containing x, y, z coordinates
    """
    min_depth = min(0.0, depth_range[1])
    max_depth = min(min_depth, depth_range[0])
    depth_range = (max_depth, min_depth)

    t = np.linspace(0, 2 * np.pi, num_points)
    x = scale * np.sin(t)
    y = scale * np.sin(t) * np.cos(t)
    z = np.linspace(depth_range[0], depth_range[1], num_points)

    headings = np.zeros(num_points)
    for i in range(num_points - 1):
        p1 = np.array([x[i], y[i], z[i]])
        p2 = np.array([x[i + 1], y[i + 1], z[i + 1]])
        headings[i] = calculate_heading(p1, p2)
    headings[-1] = headings[-2]  # Last heading same as second last

    return np.column_stack((x, y, z, headings))


def generate_square_trajectory(
    num_points=100, side_length=5.0, depth_range=(-9.0, 0.0)
):
    """
    Generate a 3D square trajectory

    Args:
        num_points: Number of waypoints to generate
        side_length: Length of each side of the square
        depth_range: Tuple of (max_depth, min_depth) for z oscillation (negative values for depth, default -9.0 to 0)

    Returns:
        numpy array of shape (num_points, 3) containing x, y, z coordinates
    """
    min_depth = min(0.0, depth_range[1])
    max_depth = min(min_depth, depth_range[0])
    depth_range = (max_depth, min_depth)

    points_per_side = num_points // 4

    # 4 line segments for the square sides
    half_len = side_length / 2

    x1 = np.linspace(-half_len, half_len, points_per_side)
    y1 = np.ones(points_per_side) * -half_len

    x2 = np.ones(points_per_side) * half_len
    y2 = np.linspace(-half_len, half_len, points_per_side)

    x3 = np.linspace(half_len, -half_len, points_per_side)
    y3 = np.ones(points_per_side) * half_len

    x4 = np.ones(points_per_side) * -half_len
    y4 = np.linspace(half_len, -half_len, points_per_side)

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])

    if len(x) < num_points:
        pad_length = num_points - len(x)
        x = np.pad(x, (0, pad_length), mode="edge")
        y = np.pad(y, (0, pad_length), mode="edge")
    elif len(x) > num_points:
        x = x[:num_points]
        y = y[:num_points]

    z = (
        depth_range[0]
        + (depth_range[1] - depth_range[0])
        * (np.sin(np.linspace(0, 2 * np.pi, num_points)) + 1)
        / 2
    )

    headings = np.zeros(num_points)
    for i in range(num_points - 1):
        p1 = np.array([x[i], y[i], z[i]])
        p2 = np.array([x[i + 1], y[i + 1], z[i + 1]])
        headings[i] = calculate_heading(p1, p2)
    headings[-1] = headings[-2]  # Last heading same as second last

    return np.column_stack((x, y, z, headings))


def generate_straight_line_trajectory(num_points=100, end_x=10.0):
    """
    Generate a 3D straight line trajectory that only moves along the x-axis from origin

    Args:
        num_points: Number of waypoints to generate
        end_x: X-coordinate of the ending point, default is 10.0

    Returns:
        numpy array of shape (num_points, 3) containing x, y, z coordinates
    """
    x = np.linspace(0, end_x, num_points)
    y = np.zeros(num_points)
    z = np.zeros(num_points)

    headings = np.zeros(num_points)
    for i in range(num_points - 1):
        p1 = np.array([x[i], y[i], z[i]])
        p2 = np.array([x[i + 1], y[i + 1], z[i + 1]])
        headings[i] = calculate_heading(p1, p2)
    headings[-1] = headings[-2]  # Last heading same as second last

    return np.column_stack((x, y, z))


def save_trajectory_to_csv(trajectory, filepath):
    """
    Save trajectory waypoints to a CSV file

    Args:
        trajectory: numpy array of shape (num_points, 4) containing x, y, z, heading (radian) coordinates
        filepath: path to save the CSV file (str or Path object)
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["x", "y", "z"])  # Write header
        writer.writerows(trajectory)

    print(f"Trajectory saved to {filepath}")


def plot_trajectory(trajectory, title="3D Trajectory"):
    """
    Plot a 3D trajectory

    Args:
        trajectory: numpy array of shape (num_points, 3) containing x, y, z coordinates
        title: plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "b-", linewidth=2)
    ax.plot(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        "go",
        markersize=10,
        label="Start",
    )
    ax.plot(
        trajectory[-1, 0],
        trajectory[-1, 1],
        trajectory[-1, 2],
        "ro",
        markersize=10,
        label="End",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.show()


class TrajectoryType(str, Enum):
    SPIRAL = "spiral"
    LEMNISCATE = "lemniscate"
    SQUARE = "square"
    STRAIGHT_LINE = "straight_line"


app = typer.Typer(help="Generate 3D trajectory waypoints and save to CSV")


@app.command()
def main(
    trajectory_type: TrajectoryType = typer.Option(
        TrajectoryType.SPIRAL, "--type", "-t", help="Type of trajectory to generate"
    ),
    points: int = typer.Option(
        100, "--points", "-p", help="Number of waypoints to generate"
    ),
    depth: float = typer.Option(
        9.0, "--depth", "-d", help="Maximum depth for the trajectory (in meters)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output CSV file path (default: auto-generated)"
    ),
    plot: bool = typer.Option(
        False, "--plot", help="Plot the trajectory after generating it"
    ),
    end_x: float = typer.Option(
        10.0, "--end-x", help="X-coordinate of the ending point (for straight line)"
    ),
):
    """Generate 3D trajectory waypoints and save to CSV."""

    if trajectory_type == TrajectoryType.SPIRAL:
        trajectory = generate_spiral_trajectory(num_points=points, depth=depth)
        traj_type = "spiral"
    elif trajectory_type == TrajectoryType.LEMNISCATE:
        trajectory = generate_lemniscate_trajectory(
            num_points=points, depth_range=(-depth, 0.0)
        )
        traj_type = "lemniscate"
    elif trajectory_type == TrajectoryType.SQUARE:
        trajectory = generate_square_trajectory(
            num_points=points, depth_range=(-depth, 0.0)
        )
        traj_type = "square"
    elif trajectory_type == TrajectoryType.STRAIGHT_LINE:
        trajectory = generate_straight_line_trajectory(num_points=points, end_x=end_x)
        traj_type = "straight_line"
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "trajectories"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{traj_type}_{timestamp}.csv"
    else:
        output_file = Path(output)

    save_trajectory_to_csv(trajectory, output_file)

    if plot:
        title = f"{trajectory_type.value.capitalize()} Trajectory ({points} points)"
        if trajectory_type == TrajectoryType.STRAIGHT_LINE:
            title = f"{trajectory_type.value.capitalize()} Trajectory: (0,0,0) to ({end_x},0,0) ({points} points)"
        elif trajectory_type in [
            TrajectoryType.SPIRAL,
            TrajectoryType.LEMNISCATE,
            TrajectoryType.SQUARE,
        ]:
            title = f"{trajectory_type.value.capitalize()} Trajectory (max depth: {depth}m, {points} points)"

        plot_trajectory(trajectory, title=title)


if __name__ == "__main__":
    app()
